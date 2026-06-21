# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers>=4.45", "peft>=0.13", "trl>=0.12", "datasets>=3.0",
#   "accelerate>=1.0", "bitsandbytes>=0.44", "huggingface_hub>=0.34",
# ]
# ///
"""QLoRA fine-tune of a small LLM on the categorization task (SCAFFOLD).

Heavier than the encoders — meant to run on a GPU via **HF Jobs**, not locally:

  hf jobs uv run --flavor a100-large --secrets HF_TOKEN \\
    training/llm_lora/train.py --dataset <ns>/transclassify-data --base Qwen/Qwen2.5-1.5B-Instruct \\
    --hf-repo <ns>/transclassify-qwen-lora

Reads the pushed HF dataset (its `train` split, exported by data.store.to_hf_datasetdict + push_to_hub),
formats each row into an instruction (taxonomy + transaction -> gold category), trains a LoRA adapter,
and pushes it. Review/tune the prompt + hyperparameters before a real run.
"""
from __future__ import annotations

import argparse


def format_example(row: dict) -> dict:
    cats = "\n".join(f"- {c['name']}" + (f": {c['description']}" if c.get("description") else "")
                     for c in row["categories"])
    prompt = (
        "Categorize the transaction into exactly one of the allowed categories.\n\n"
        f"Allowed categories:\n{cats}\n\nTransaction: {row['description']}\n\nCategory:"
    )
    return {"prompt": prompt, "completion": " " + row["gold"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset id (has a 'train' split)")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--hf-repo", required=True, help="where to push the LoRA adapter")
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset(args.dataset, split="train").map(format_example)
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")
    trainer = SFTTrainer(
        model=args.base,
        train_dataset=ds,
        peft_config=peft_config,
        args=SFTConfig(
            output_dir="out",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=8,
            learning_rate=2e-4,
            bf16=True,
            push_to_hub=True,
            hub_model_id=args.hf_repo,
        ),
    )
    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
