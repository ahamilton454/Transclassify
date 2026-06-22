"""Hand-run the served Ollama model on ONE example and print its RAW output.

No Instructor / no strict schema — just the prompt the model was *trained* on
(see mlx_make_data.py) and whatever text the model returns. Use it to see what's
actually coming back vs. what the strict harness expects.

  ollama serve            # (if not already running)
  backend/.venv/bin/python training/llm_lora/try_one.py
  backend/.venv/bin/python training/llm_lora/try_one.py --i 7
  backend/.venv/bin/python training/llm_lora/try_one.py --desc "SQ *PERK UP COFFEE PORTLAND OR"
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.store import load  # noqa: E402


def render_categories(categories) -> str:
    lines = []
    for c in categories:
        label = f"{c.parent} > {c.name}" if c.parent else c.name
        lines.append(f"- {label}" + (f": {c.description}" if c.description else ""))
    return "\n".join(lines)


def training_prompt(categories, description: str) -> str:
    # Mirrors training/llm_lora/mlx_make_data.py exactly.
    return (
        "Categorize the transaction into exactly one of the allowed categories. "
        "Respond with only the category name.\n\n"
        f"Allowed categories:\n{render_categories(categories)}\n\n"
        f"Transaction: {description}"
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ollama_chat/transclassify-qwen")
    ap.add_argument("--i", type=int, default=0, help="which eval example index")
    ap.add_argument("--desc", default=None, help="ad-hoc description (overrides --i's text)")
    ap.add_argument("--set", default="llm_generated")
    args = ap.parse_args()

    records = load(split="eval", source=args.set)
    if not records:
        raise SystemExit(f"no eval data for source {args.set!r}")
    rec = records[args.i]
    description = args.desc or rec.transaction.description
    prompt = training_prompt(rec.categories, description)

    print("=" * 70)
    print("PROMPT (as trained):\n")
    print(prompt)
    print("=" * 70)
    print(f"GOLD: {rec.gold}    (allowed: {[c.name for c in rec.categories]})")
    print("=" * 70)

    from litellm import acompletion

    resp = await acompletion(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content
    print(f"RAW MODEL OUTPUT:\n{raw!r}")
    print("=" * 70)
    exact = raw.strip() in {c.name for c in rec.categories}
    ci = raw.strip().lower() in {c.name.lower() for c in rec.categories}
    print(f"exact-match allowed? {exact}   case-insensitive match? {ci}")


if __name__ == "__main__":
    asyncio.run(main())
