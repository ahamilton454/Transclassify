"""Export the train split to MLX-LM chat format for local LoRA on Apple Silicon.

Writes training/llm_lora/mlx_data/{train,valid}.jsonl as chat messages
(user = categorize prompt, assistant = gold category), which `mlx_lm.lora` consumes.

  backend/.venv/bin/python training/llm_lora/mlx_make_data.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.store import load  # noqa: E402

OUT = Path(__file__).resolve().parent / "mlx_data"


def render_categories(categories) -> str:
    lines = []
    for c in categories:
        label = f"{c.parent} > {c.name}" if c.parent else c.name
        lines.append(f"- {label}" + (f": {c.description}" if c.description else ""))
    return "\n".join(lines)


def to_row(ex) -> dict:
    prompt = (
        "Categorize the transaction into exactly one of the allowed categories. "
        "Respond with only the category name.\n\n"
        f"Allowed categories:\n{render_categories(ex.categories)}\n\n"
        f"Transaction: {ex.transaction.description}"
    )
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ex.gold},
        ]
    }


def main() -> None:
    examples = load(split="train")
    if not examples:
        raise SystemExit(
            "no train data — run: data/generate.py --source synthetic_v2 --split train --total 10000"
        )
    rows = [to_row(e) for e in examples]
    random.Random(0).shuffle(rows)
    n_valid = max(1, len(rows) // 10)
    valid, train = rows[:n_valid], rows[n_valid:]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
    (OUT / "valid.jsonl").write_text("\n".join(json.dumps(r) for r in valid) + "\n")
    print(f"wrote {len(train)} train + {len(valid)} valid -> {OUT}")


if __name__ == "__main__":
    main()
