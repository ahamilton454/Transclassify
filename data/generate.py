"""Hybrid generator for eval + training data (realistic surface form, trustworthy gold).

Two layers (see data/transaction_patterns.md):
  1. An LLM supplies category-appropriate MERCHANTS/payees and tags each with a
     statement `channel` (in_store, online, ach_biller, payroll, p2p, fee, ...).
  2. The deterministic template engine (data/templates.py) wraps each merchant in a
     realistic bank/processor/locator/ref line for that channel.

The merchant fixes the gold label (we asked for category C), and the engine — not the
LLM — produces the gnarly statement formatting, so we get realistic strings AND
reliable labels with far more format diversity than asking an LLM for whole strings.

  # eval set across all taxonomies (incl. held-out):
  python data/generate.py --source synthetic_v2 --split eval --total 2000 \
      --taxonomies personal_budget freelancer_schedule_c small_business_coa \
                   student_budget rental_property nonprofit_org \
                   restaurant_owner ecommerce_seller
  # training set across the trained taxonomies only, disjoint from eval:
  python data/generate.py --source synthetic_v2 --split train --total 10000 \
      --taxonomies personal_budget freelancer_schedule_c small_business_coa \
                   student_budget rental_property nonprofit_org

CAVEAT: gold is a strong LLM's choice of merchant→category, not human ground truth.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import random
import re
import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pydantic import BaseModel, Field, create_model  # noqa: E402

from app.config import settings  # noqa: E402 — loads .env + exports key

from models.llm_incontext import engine  # noqa: E402

from data import templates  # noqa: E402
from data.schema import Labeling  # noqa: E402
from data.store import append, load, load_category_sets, make_transaction  # noqa: E402

TRAINED_TAXONOMIES = [
    "personal_budget", "freelancer_schedule_c", "small_business_coa",
    "student_budget", "rental_property", "nonprofit_org",
]
HELD_OUT_TAXONOMIES = ["restaurant_owner", "ecommerce_seller"]
PREFIX_RE = re.compile(r"[A-Z]{2,4}\s*\*|\bPOS\b|\bACH\b|\bSQ\b|TST\*|\bPPD\b|\bEFT\b|\bPYPL\b", re.I)

CHANNEL_HINTS = {
    "in_store": "card-present retail/grocery/pharmacy",
    "online": "online / card-not-present purchase",
    "subscription": "recurring SaaS, streaming, membership",
    "gas": "fuel at a gas station",
    "restaurant": "dining, bar, takeout",
    "ach_biller": "recurring biller debit (utilities, insurance, rent, loan, mortgage)",
    "payroll": "incoming payroll / direct deposit (a credit)",
    "p2p": "Venmo/Zelle/Cash App/PayPal person-to-person",
    "atm_cash": "ATM cash withdrawal",
    "bank_fee": "bank or card-processing fee",
    "government": "tax, DMV, or municipal payment",
    "check": "paper check or mobile deposit",
    "deposit": "sales-revenue or card-batch deposit (a credit)",
}
ChannelLiteral = Literal[tuple(templates.CHANNELS)]  # type: ignore[valid-type]

SYSTEM = (
    "You build a labeled dataset of US bank/card transactions. For a given category you "
    "name realistic merchants/payees/counterparties that UNAMBIGUOUSLY belong to that "
    "category, and tag how each would reach the statement (its channel). You do not write "
    "the final statement string — only the merchant and channel."
)


class MerchantEntry(BaseModel):
    name: str = Field(description="Real US merchant, payee, employer, agency, or person name.")
    channel: ChannelLiteral = Field(description="How this transaction appears on a statement.")


def _entries_model(k: int) -> type[BaseModel]:
    return create_model(
        "MerchantList",
        merchants=(list[MerchantEntry], Field(description=f"{k} distinct entries for the category.")),
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


async def _merchants_for_category(model, taxonomy, cat, other_names, k) -> list[MerchantEntry]:
    desc = f" ({cat.description})" if cat.description else ""
    channels = "\n".join(f"  - {c}: {h}" for c, h in CHANNEL_HINTS.items())
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Taxonomy: {taxonomy}.\n"
                f"Category: '{cat.name}'{desc}.\n\n"
                f"Give {k} DISTINCT merchants/payees whose transactions clearly belong to "
                f"'{cat.name}' and NOT to any of: {', '.join(other_names)}.\n"
                "Mix well-known national names with small/local/obscure ones. For each, choose "
                "the most realistic channel from:\n"
                f"{channels}\n"
                "Use credit channels (payroll, deposit) only for incoming-money categories."
            ),
        },
    ]
    try:
        out, _ = await engine.complete(model, messages, _entries_model(k))
        seen, result = set(), []
        for e in out.merchants:
            key = (e.name.strip().lower(), e.channel)
            if e.name.strip() and key not in seen:
                seen.add(key)
                result.append(e)
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {taxonomy}/{cat.name}: {exc}")
        return []


def _seed(source: str, split: str) -> int:
    return int.from_bytes(hashlib.sha1(f"{source}:{split}".encode()).digest()[:4], "big")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--split", default="eval", choices=["train", "eval"])
    ap.add_argument("--total", type=int, default=2000)
    ap.add_argument("--taxonomies", nargs="*", default=TRAINED_TAXONOMIES)
    ap.add_argument("--per-category-merchants", type=int, default=30)
    ap.add_argument("--model", default="openai/gpt-5-mini")
    args = ap.parse_args()

    registry = load_category_sets()
    per_tax = args.total // len(args.taxonomies)

    # 1) LLM: a merchant pool per (taxonomy, category).
    jobs = []
    for tax_id in args.taxonomies:
        cs = registry[tax_id]
        names = [c.name for c in cs.categories]
        for cat in cs.categories:
            jobs.append((tax_id, cat, [n for n in names if n != cat.name]))
    print(f"[{args.split}] fetching merchant pools: {len(jobs)} categories x "
          f"{args.per_category_merchants} via {args.model} ...")
    pools = await engine.gather_bounded(
        [_merchants_for_category(args.model, t, c, o, args.per_category_merchants) for t, c, o in jobs],
        concurrency=8,
    )

    # 2) Templates: compose balanced, deduped transactions per category.
    other_split = "eval" if args.split == "train" else "train"
    forbidden = {ex.transaction.id for ex in load(split=other_split, source=args.source)}
    rng = random.Random(_seed(args.source, args.split))

    txns, labs, seen = [], [], set()
    skipped = 0
    for (tax_id, cat, _others), entries in zip(jobs, pools):
        if not entries:
            continue
        ncats = len(registry[tax_id].categories)
        target = max(1, per_tax // ncats)
        made, attempts = 0, 0
        while made < target and attempts < target * 8:
            attempts += 1
            e = rng.choice(entries)
            desc = templates.compose(e.name, e.channel, rng)
            tx = make_transaction(desc, amount=templates.amount_for(e.channel, rng))
            if tx.id in forbidden or tx.id in seen:
                skipped += 1
                continue
            seen.add(tx.id)
            txns.append(tx)
            labs.append(Labeling(
                transaction_id=tx.id, category_set_id=tax_id, gold=cat.name,
                strata=_strata(desc), split=args.split, source=args.source,
            ))
            made += 1

    append(args.source, txns, labs)
    print(f"[{args.split}] appended {len(labs)} labelings to data/sources/{args.source}/ "
          f"({skipped} dup/disjoint skipped)")


if __name__ == "__main__":
    asyncio.run(main())
