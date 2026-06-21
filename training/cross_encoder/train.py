"""Fine-tune a cross-encoder (reranker) as a pair classifier on the train split.

For each example: the (description, gold category text) pair is a positive (label 1);
a few other categories from the same taxonomy are negatives (label 0).

  backend/.venv/bin/python training/cross_encoder/train.py [--epochs 1] [--hf-repo ns/name]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.store import load  # noqa: E402
from models.encoder_common import category_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="BAAI/bge-reranker-base")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--negatives", type=int, default=3)
    ap.add_argument("--out", default=str(Path(__file__).parent / "output"))
    ap.add_argument("--hf-repo", default=None)
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    examples = load(split="train")
    if not examples:
        raise SystemExit(
            "no train data — run: data/generate.py --source training_v1 --split train --total 5000"
        )
    rng = random.Random(0)
    samples: list[InputExample] = []
    for ex in examples:
        gold = next(c for c in ex.categories if c.name == ex.gold)
        samples.append(InputExample(texts=[ex.transaction.description, category_text(gold)], label=1.0))
        negs = [c for c in ex.categories if c.name != ex.gold]
        for neg in rng.sample(negs, min(args.negatives, len(negs))):
            samples.append(
                InputExample(texts=[ex.transaction.description, category_text(neg)], label=0.0)
            )
    print(f"cross-encoder fine-tune: {len(samples)} pairs, base={args.base}, epochs={args.epochs}")

    model = CrossEncoder(args.base, num_labels=1)
    loader = DataLoader(samples, shuffle=True, batch_size=args.batch_size)
    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=int(len(loader) * args.epochs * 0.1),
        show_progress_bar=True,
    )
    model.save(args.out)
    print(f"saved -> {args.out}")
    if args.hf_repo:
        model.push_to_hub(args.hf_repo)
        print(f"pushed -> {args.hf_repo}")


if __name__ == "__main__":
    main()
