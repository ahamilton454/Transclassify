"""Fetch + sample + map the `DoDataThings/us-bank-transaction-categories-v2`
dataset (Hugging Face, public/ungated) into the eval record schema.

Writes:
  - evals/dodatathings/data.jsonl                  (sampled records, gitignored)
  - evals/category_sets/dodatathings_us_v2.json    (the dataset's fixed label space)

Usage:  backend/.venv/bin/python evals/dodatathings/prepare.py --limit 200

This is the *external, fixed-taxonomy* eval — a regression signal and an external
number, NOT the trusted decision-maker. It's a community upload (realistic
sign-prefixed US bank strings, ~17 categories), so its gold is someone else's
labels — audit a sample before trusting. Streaming + first-N for reproducibility.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
CATEGORY_SETS_DIR = ROOT / "evals" / "category_sets"
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
    # crude long-tail heuristic: digit-heavy / star-coded => messy
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

    # Peek one row to discover column names, then re-open (streaming is consumed).
    first = next(iter(_open()))
    desc_key = _pick(list(first.keys()), DESC_KEYS)
    label_key = _pick(list(first.keys()), LABEL_KEYS)
    print(f"mapping: description<-{desc_key!r}  gold<-{label_key!r}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels: set[str] = set()
    rows = []
    for i, row in enumerate(_open()):
        if i >= args.limit:
            break
        desc = str(row[desc_key]).strip()
        gold = str(row[label_key]).strip()
        if not desc or not gold:
            continue
        labels.add(gold)
        rows.append(
            {
                "id": f"dd_{i:05d}",
                "transaction": {"id": f"dd_{i:05d}", "description": desc},
                "category_set_id": CATEGORY_SET_ID,
                "gold": gold,
                "strata": _strata(desc),
            }
        )

    # Write the discovered label space as a category set (no descriptions).
    cat_set = {"id": CATEGORY_SET_ID, "categories": [{"name": n} for n in sorted(labels)]}
    (CATEGORY_SETS_DIR / f"{CATEGORY_SET_ID}.json").write_text(json.dumps(cat_set, indent=2))

    data_path = OUT_DIR / "data.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} records to {data_path} across {len(labels)} categories")


if __name__ == "__main__":
    sys.exit(main())
