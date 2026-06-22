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

    from datasets import Dataset
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    examples = load(split="train")
    if not examples:
        raise SystemExit(
            "no train data — run: data/generate.py --source synthetic_v2 --split train --total 10000"
        )
    rng = random.Random(0)
    q, p, y = [], [], []
    for ex in examples:
        gold = next(c for c in ex.categories if c.name == ex.gold)
        q.append(ex.transaction.description); p.append(category_text(gold)); y.append(1.0)
        negs = [c for c in ex.categories if c.name != ex.gold]
        for neg in rng.sample(negs, min(args.negatives, len(negs))):
            q.append(ex.transaction.description); p.append(category_text(neg)); y.append(0.0)
    ds = Dataset.from_dict({"query": q, "passage": p, "label": y})
    print(f"cross-encoder fine-tune: {len(ds)} pairs, base={args.base}, epochs={args.epochs}")

    model = CrossEncoder(args.base, num_labels=1)
    trainer = CrossEncoderTrainer(
        model=model,
        args=CrossEncoderTrainingArguments(
            # checkpoints go to a separate dir; the final model is saved cleanly to args.out
            output_dir=args.out + "_ckpt",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            warmup_ratio=0.1,
            report_to=[],
            logging_strategy="no",
            save_strategy="no",
        ),
        train_dataset=ds,
        loss=BinaryCrossEntropyLoss(model),
    )
    trainer.train()
    model.save_pretrained(args.out)
    print(f"saved -> {args.out}")
    if args.hf_repo:
        model.push_to_hub(args.hf_repo)
        print(f"pushed -> {args.hf_repo}")


if __name__ == "__main__":
    main()
