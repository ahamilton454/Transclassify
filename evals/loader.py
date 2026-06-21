"""Load + validate eval data: the category-set registry and JSONL records.

Resolves each record's categories (registry reference or inline override) and
validates that gold/acceptable are actually in the taxonomy, so a malformed record
fails loudly here rather than silently scoring as wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

from evals.schema import CategorySet, EvalRecord

EVALS_DIR = Path(__file__).resolve().parent
CATEGORY_SETS_DIR = EVALS_DIR / "category_sets"


def load_category_sets(directory: Path = CATEGORY_SETS_DIR) -> dict[str, CategorySet]:
    sets: dict[str, CategorySet] = {}
    for path in sorted(directory.glob("*.json")):
        cs = CategorySet.model_validate_json(path.read_text())
        sets[cs.id] = cs
    return sets


def load_records(jsonl_path: Path, registry: dict[str, CategorySet]) -> list[EvalRecord]:
    """Parse a JSONL file into validated records with categories resolved in place."""
    records: list[EvalRecord] = []
    for lineno, line in enumerate(Path(jsonl_path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = EvalRecord.model_validate_json(line)
            _resolve_and_validate(rec, registry)
        except Exception as exc:  # noqa: BLE001 — annotate which line broke
            raise ValueError(f"{jsonl_path}:{lineno} — {exc}") from exc
        records.append(rec)
    return records


def _resolve_and_validate(rec: EvalRecord, registry: dict[str, CategorySet]) -> None:
    if rec.categories is None:
        if rec.category_set_id is None:
            raise ValueError(f"record {rec.id}: needs category_set_id or inline categories")
        cs = registry.get(rec.category_set_id)
        if cs is None:
            raise ValueError(f"record {rec.id}: unknown category_set_id {rec.category_set_id!r}")
        rec.categories = cs.categories

    names = {c.name for c in rec.categories}
    if rec.gold not in names:
        raise ValueError(f"record {rec.id}: gold {rec.gold!r} not in category set {sorted(names)}")
    for a in rec.resolved_acceptable():
        if a not in names:
            raise ValueError(f"record {rec.id}: acceptable {a!r} not in category set {sorted(names)}")
