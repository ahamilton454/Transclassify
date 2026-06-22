"""Contrastive fine-tune of a bi-encoder on the train split, then save (+ optional HF push).

Learns to embed a transaction near its gold category's text (the same `category_text`
rendering the inference strategy uses). Multi-taxonomy training → BYO generalization.

  backend/.venv/bin/python training/bi_encoder/train.py [--epochs 1] [--hf-repo ns/name]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.store import load  # noqa: E402
from models.encoder_common import category_text  # noqa: E402


def _gold_category(example):
    return next(c for c in example.categories if c.name == example.gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=str(Path(__file__).parent / "output"))
    ap.add_argument("--hf-repo", default=None)
    args = ap.parse_args()

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    examples = load(split="train")
    if not examples:
        raise SystemExit(
            "no train data — run: data/generate.py --source synthetic_v2 --split train --total 10000"
        )
    pairs = [
        InputExample(texts=[ex.transaction.description, category_text(_gold_category(ex))])
        for ex in examples
    ]
    print(f"bi-encoder fine-tune: {len(pairs)} pairs, base={args.base}, epochs={args.epochs}")

    model = SentenceTransformer(args.base)
    loader = DataLoader(pairs, shuffle=True, batch_size=args.batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)  # in-batch negatives
    model.fit(
        train_objectives=[(loader, loss)],
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
