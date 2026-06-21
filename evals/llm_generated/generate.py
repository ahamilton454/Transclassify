"""Generate a varied synthetic eval set across the representative taxonomies.

Approach (see also the caveats below):
- Balanced per category: for each taxonomy, for each category, ask a strong LLM
  (gpt-5.2 by default) to produce K varied transaction strings that belong to that
  category. gold = that category. This guarantees per-category coverage and grounds
  the gold in the generator's intent.
- The LLM is given the *other* category names so it avoids cross-category overlap.

CAVEATS (read before trusting numbers):
- The gold is a strong LLM's label, not human ground truth. So LLM categorizers
  (especially same-family ones) have a home-field advantage on this set — it measures
  "how close does each categorizer get to gpt-5.2's labels." Encoder / small-local gaps
  and per-category differences are still informative.

Usage:  backend/.venv/bin/python evals/llm_generated/generate.py --total 1000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pydantic import BaseModel, Field, create_model  # noqa: E402

from app.config import settings  # noqa: E402 — loads .env + exports key
from models.llm_incontext import engine  # noqa: E402

from evals.loader import load_category_sets  # noqa: E402

OUT = Path(__file__).resolve().parent / "data.jsonl"
TAXONOMIES = ["personal_budget", "freelancer_schedule_c", "small_business_coa"]
PREFIX_RE = re.compile(r"[A-Z]{2,4}\s*\*|\bPOS\b|\bACH\b|\bSQ\b|TST\*|\bPPD\b|\bEFT\b|\bPYPL\b", re.I)

SYSTEM = (
    "You generate realistic, varied US bank and credit-card transaction description strings for an "
    "evaluation dataset. Output only raw transaction strings exactly as they'd appear on a statement "
    "— no explanations, no numbering."
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


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


async def gen_category(model: str, taxonomy: str, cat, other_names: list[str], k: int) -> list[str]:
    desc = f" ({cat.description})" if cat.description else ""
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Taxonomy: {taxonomy}.\n"
                f"Generate {k} DISTINCT, realistic transaction description strings that clearly belong "
                f"to the category '{cat.name}'{desc}.\n"
                "Vary them widely: mix well-known national merchants with small/local/obscure ones; "
                "vary formats — clean names, processor prefixes (SQ *, TST*, PYPL*, PAYPAL *), P2P "
                "(Venmo/Zelle), ACH/PPD/EFT, gas-pump and store-number noise, and a few genuinely "
                "cryptic ones.\n"
                f"Each must fit '{cat.name}' better than any of these other categories: "
                f"{', '.join(other_names)}."
            ),
        },
    ]
    try:
        out, _ = await engine.complete(model, messages, _examples_model(k))
        # dedupe, keep order
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
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--model", default="openai/gpt-5.2")
    args = ap.parse_args()

    registry = load_category_sets()
    per_tax = args.total // len(TAXONOMIES)

    jobs = []  # (taxonomy_id, category, other_names, k)
    for tax_id in TAXONOMIES:
        cs = registry[tax_id]
        k = max(1, round(per_tax / len(cs.categories)))
        names = [c.name for c in cs.categories]
        for cat in cs.categories:
            others = [n for n in names if n != cat.name]
            jobs.append((tax_id, cat, others, k))

    print(f"generating ~{args.total} examples across {len(TAXONOMIES)} taxonomies "
          f"({len(jobs)} category jobs) with {args.model} ...")
    batches = await engine.gather_bounded(
        [gen_category(args.model, t, c, o, k) for (t, c, o, k) in jobs], concurrency=8
    )

    records, idx = [], 0
    for (tax_id, cat, _others, _k), descs in zip(jobs, batches):
        for d in descs:
            rid = f"{tax_id[:3]}_{_slug(cat.name)}_{idx:05d}"
            records.append(
                {
                    "id": rid,
                    "transaction": {"id": rid, "description": d},
                    "category_set_id": tax_id,
                    "gold": cat.name,
                    "strata": _strata(d),
                }
            )
            idx += 1

    OUT.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    by_tax = {t: sum(1 for r in records if r["category_set_id"] == t) for t in TAXONOMIES}
    print(f"wrote {len(records)} records to {OUT}")
    print("per taxonomy:", by_tax)


if __name__ == "__main__":
    asyncio.run(main())
