"""Read/write the normalized dataset and query it.

Source of truth = per-source JSONL (`data/sources/<source>/{transactions,labelings}.jsonl`)
+ the shared `data/category_sets/`. `load()` joins them into `Example`s; DuckDB
answers analytical queries over the JSONL in place (no separate store to sync);
`to_hf_datasetdict()` exports train/eval splits for training.
"""
from __future__ import annotations

import json
from pathlib import Path

from models.types import Transaction

from data.schema import CategorySet, Example, Labeling, transaction_id

DATA_DIR = Path(__file__).resolve().parent
CATEGORY_SETS_DIR = DATA_DIR / "category_sets"
SOURCES_DIR = DATA_DIR / "sources"


# --------------------------------------------------------------------------- #
# Category sets
# --------------------------------------------------------------------------- #
def load_category_sets(directory: Path | None = None) -> dict[str, CategorySet]:
    directory = directory or CATEGORY_SETS_DIR  # read module global at call time (testable)
    sets: dict[str, CategorySet] = {}
    for path in sorted(directory.glob("*.json")):
        cs = CategorySet.model_validate_json(path.read_text())
        sets[cs.id] = cs
    return sets


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _source_dirs(source: str | None) -> list[Path]:
    if source:
        return [SOURCES_DIR / source]
    return [p for p in sorted(SOURCES_DIR.glob("*")) if p.is_dir()] if SOURCES_DIR.exists() else []


def load(split: str | None = None, source: str | None = None) -> list[Example]:
    """Join labelings → transactions + resolved categories, filtered by split/source."""
    registry = load_category_sets()
    examples: list[Example] = []
    for sdir in _source_dirs(source):
        transactions = {
            t["id"]: Transaction.model_validate(t)
            for t in _read_jsonl(sdir / "transactions.jsonl")
        }
        for raw in _read_jsonl(sdir / "labelings.jsonl"):
            lab = Labeling.model_validate(raw)
            if split and lab.split != split:
                continue
            tx = transactions.get(lab.transaction_id)
            cs = registry.get(lab.category_set_id)
            if tx is None or cs is None:
                raise ValueError(
                    f"{sdir.name}: labeling refers to missing "
                    f"{'transaction ' + lab.transaction_id if tx is None else 'category_set ' + lab.category_set_id}"
                )
            names = cs.names()
            for label in {lab.gold, *(lab.acceptable or [])}:
                if label not in names:
                    raise ValueError(
                        f"{sdir.name}: {label!r} not in category set {lab.category_set_id!r} {sorted(names)}"
                    )
            examples.append(
                Example(
                    transaction=tx,
                    categories=cs.categories,
                    category_set_id=lab.category_set_id,
                    gold=lab.gold,
                    acceptable=lab.acceptable,
                    expected_other=lab.expected_other,
                    pair_id=lab.pair_id,
                    strata=lab.strata,
                    note=lab.note,
                    split=lab.split,
                    source=lab.source or sdir.name,
                )
            )
    return examples


# --------------------------------------------------------------------------- #
# Write (append + dedup)
# --------------------------------------------------------------------------- #
def append(source: str, transactions: list[Transaction], labelings: list[Labeling]) -> None:
    """Merge transactions (dedup by id) and append labelings for a source."""
    sdir = SOURCES_DIR / source
    sdir.mkdir(parents=True, exist_ok=True)

    tx_path = sdir / "transactions.jsonl"
    existing = {t["id"]: t for t in _read_jsonl(tx_path)}
    for t in transactions:
        existing.setdefault(t.id, t.model_dump())
    tx_path.write_text("\n".join(json.dumps(t) for t in existing.values()) + "\n")

    lab_path = sdir / "labelings.jsonl"
    lines = [json.dumps(lab.model_dump()) for lab in labelings]
    with lab_path.open("a") as f:
        for line in lines:
            f.write(line + "\n")


def make_transaction(description: str, **fields) -> Transaction:
    """Build a pooled Transaction with a content-hash id."""
    return Transaction(id=transaction_id(description), description=description, **fields)


# --------------------------------------------------------------------------- #
# DuckDB queries (analytics over the JSONL, in place)
# --------------------------------------------------------------------------- #
def _labelings_glob() -> str:
    return str(SOURCES_DIR / "*" / "labelings.jsonl")


def overlap_transaction_ids(split_a: str = "train", split_b: str = "eval") -> list[str]:
    """Transaction ids carrying labelings in BOTH splits (must be empty to stay honest)."""
    import duckdb

    rows = duckdb.sql(
        f"""
        SELECT transaction_id
        FROM read_json_auto('{_labelings_glob()}', union_by_name=true)
        WHERE split IN ('{split_a}', '{split_b}')
        GROUP BY transaction_id
        HAVING count(DISTINCT split) > 1
        """
    ).fetchall()
    return [r[0] for r in rows]


def counts_by(field: str = "split") -> list[tuple]:
    import duckdb

    return duckdb.sql(
        f"SELECT {field}, count(*) FROM read_json_auto('{_labelings_glob()}', union_by_name=true) "
        f"GROUP BY {field} ORDER BY 2 DESC"
    ).fetchall()


# --------------------------------------------------------------------------- #
# HF export (for training)
# --------------------------------------------------------------------------- #
def to_hf_datasetdict(splits: tuple[str, ...] = ("train", "eval")):
    """Export splits as a HF DatasetDict (lazy import; only needed for training)."""
    from datasets import Dataset, DatasetDict

    out = {}
    for split in splits:
        rows = [
            {
                "description": ex.transaction.description,
                "category_set_id": ex.category_set_id,
                "categories": [c.model_dump() for c in ex.categories],
                "gold": ex.gold,
                "source": ex.source,
            }
            for ex in load(split=split)
        ]
        out[split] = Dataset.from_list(rows)
    return DatasetDict(out)
