"""The single LLM generator for both eval and training data.

Generates balanced, varied transaction strings per category across taxonomies,
labeled by a strong LLM (gpt-5.2), and appends them to the pool tagged with a
`(source, split)`. Generating training data dedups against the eval split (by
transaction id) so train ∩ eval = ∅.

  # eval set (≈ the old evals/llm_generated/generate.py):
  python data/generate.py --source llm_generated --split eval --total 1000
  # training corpus, disjoint from eval:
  python data/generate.py --source training_v1 --split train --total 5000

CAVEAT: gold is a strong LLM's label, not human ground truth — so LLM categorizers
(esp. same-family) have a home-field advantage. Encoder/local gaps + per-category
differences are still informative.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pydantic import BaseModel, Field, create_model  # noqa: E402

from app.config import settings  # noqa: E402 — loads .env + exports key
from models.llm_incontext import engine  # noqa: E402

from data.schema import Labeling  # noqa: E402
from data.store import append, load, load_category_sets, make_transaction  # noqa: E402

DEFAULT_TAXONOMIES = ["personal_budget", "freelancer_schedule_c", "small_business_coa"]
PREFIX_RE = re.compile(r"[A-Z]{2,4}\s*\*|\bPOS\b|\bACH\b|\bSQ\b|TST\*|\bPPD\b|\bEFT\b|\bPYPL\b", re.I)

SYSTEM = (
    "You generate realistic, varied US bank and credit-card transaction description strings for an "
    "evaluation dataset. Output only raw transaction strings exactly as they'd appear on a statement "
    "— no explanations, no numbering."
)


def _strata(desc: str) -> list[str]:
    tags = []
    if PREFIX_RE.search(desc):
        tags.append("processor-prefix")
    if sum(ch.isdigit() for ch in desc) >= 4 or "*" in desc:
        tags.append("tail")
    if not tags:
        tags.append("head")
    return tags


def _examples_model(k: int) -> type[BaseModel]:
    return create_model(
        "GeneratedExamples",
        descriptions=(list[str], Field(description=f"{k} distinct transaction strings.")),
    )


async def _gen_category(model, taxonomy, cat, other_names, k) -> list[str]:
    desc = f" ({cat.description})" if cat.description else ""
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Taxonomy: {taxonomy}.\n"
                f"Generate {k} DISTINCT, realistic transaction description strings that clearly belong "
                f"to the category '{cat.name}'{desc}.\n"
                "Vary widely: mix well-known national merchants with small/local/obscure ones; vary "
                "formats — clean names, processor prefixes (SQ *, TST*, PYPL*, PAYPAL *), P2P "
                "(Venmo/Zelle), ACH/PPD/EFT, gas-pump and store-number noise, and a few cryptic ones.\n"
                f"Each must fit '{cat.name}' better than any of these: {', '.join(other_names)}."
            ),
        },
    ]
    try:
        out, _ = await engine.complete(model, messages, _examples_model(k))
        seen, result = set(), []
        for d in out.descriptions:
            d = d.strip()
            if d and d.lower() not in seen:
                seen.add(d.lower())
                result.append(d)
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {taxonomy}/{cat.name}: {exc}")
        return []


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--split", default="eval", choices=["train", "eval"])
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--taxonomies", nargs="*", default=DEFAULT_TAXONOMIES)
    ap.add_argument("--model", default="openai/gpt-5.2")
    args = ap.parse_args()

    registry = load_category_sets()
    per_tax = args.total // len(args.taxonomies)

    jobs = []
    for tax_id in args.taxonomies:
        cs = registry[tax_id]
        k = max(1, round(per_tax / len(cs.categories)))
        names = [c.name for c in cs.categories]
        for cat in cs.categories:
            jobs.append((tax_id, cat, [n for n in names if n != cat.name], k))

    print(f"generating ~{args.total} for source={args.source} split={args.split} "
          f"({len(jobs)} jobs) with {args.model} ...")
    batches = await engine.gather_bounded(
        [_gen_category(args.model, t, c, o, k) for (t, c, o, k) in jobs], concurrency=8
    )

    # Disjointness: never reuse a transaction string that exists in the opposite split.
    other = "eval" if args.split == "train" else "train"
    forbidden = {ex.transaction.id for ex in load(split=other)}

    txns, labs, seen = [], [], set()
    for (tax_id, cat, _o, _k), descs in zip(jobs, batches):
        for d in descs:
            tx = make_transaction(d)
            if tx.id in forbidden or tx.id in seen:
                continue
            seen.add(tx.id)
            txns.append(tx)
            labs.append(
                Labeling(
                    transaction_id=tx.id,
                    category_set_id=tax_id,
                    gold=cat.name,
                    strata=_strata(d),
                    split=args.split,
                    source=args.source,
                )
            )

    append(args.source, txns, labs)
    print(f"appended {len(labs)} labelings to data/sources/{args.source}/ (dropped {len(seen) and ''}"
          f"{sum(len(b) for b in batches) - len(labs)} dup/disjoint)")


if __name__ == "__main__":
    asyncio.run(main())
