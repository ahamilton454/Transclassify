"""Fetch + sample the public `DoDataThings/us-bank-transaction-categories-v2` dataset
into the pool as the `dodatathings` eval source (split=eval).

Writes labelings/transactions under data/sources/dodatathings/ and the discovered
label space to data/category_sets/dodatathings_us_v2.json. External, fixed-taxonomy,
community-labeled — a regression signal, not the trusted set (audit before trusting).

  python data/fetch_dodatathings.py --limit 200
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.schema import Labeling  # noqa: E402
from data.store import CATEGORY_SETS_DIR, append, make_transaction  # noqa: E402

DATASET = "DoDataThings/us-bank-transaction-categories-v2"
CATEGORY_SET_ID = "dodatathings_us_v2"
DESC_KEYS = ["description", "transaction", "text", "narration", "memo", "raw", "name"]
LABEL_KEYS = ["category", "label", "class", "target"]
PREFIX_RE = re.compile(r"[A-Z]{2,4}\s*\*|\bPOS\b|\bACH\b|\bSQ\b|TST\*|\bPPD\b|\bEFT\b", re.I)


def _pick(colnames: list[str], candidates: list[str]) -> str:
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        for lc, original in lower.items():
            if cand == lc or cand in lc:
                return original
    raise SystemExit(f"could not find a column among {candidates} in {colnames}")


def _strata(desc: str) -> list[str]:
    tags = ["external"]
    if PREFIX_RE.search(desc):
        tags.append("processor-prefix")
    if sum(ch.isdigit() for ch in desc) >= 4 or "*" in desc:
        tags.append("tail")
    return tags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets (it's in backend's dev extras).")

    def _open():
        try:
            return load_dataset(DATASET, split="train", streaming=True)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Could not load {DATASET!r}: {exc}") from exc

    first = next(iter(_open()))
    desc_key = _pick(list(first.keys()), DESC_KEYS)
    label_key = _pick(list(first.keys()), LABEL_KEYS)
    print(f"mapping: description<-{desc_key!r}  gold<-{label_key!r}")

    labels: set[str] = set()
    txns, labs = [], []
    for i, row in enumerate(_open()):
        if i >= args.limit:
            break
        desc = str(row[desc_key]).strip()
        gold = str(row[label_key]).strip()
        if not desc or not gold:
            continue
        labels.add(gold)
        tx = make_transaction(desc)
        txns.append(tx)
        labs.append(
            Labeling(
                transaction_id=tx.id,
                category_set_id=CATEGORY_SET_ID,
                gold=gold,
                strata=_strata(desc),
                split="eval",
                source="dodatathings",
            )
        )

    cat_set = {"id": CATEGORY_SET_ID, "categories": [{"name": n} for n in sorted(labels)]}
    (CATEGORY_SETS_DIR / f"{CATEGORY_SET_ID}.json").write_text(json.dumps(cat_set, indent=2))
    append("dodatathings", txns, labs)
    print(f"appended {len(labs)} dodatathings labelings across {len(labels)} categories")


if __name__ == "__main__":
    main()
