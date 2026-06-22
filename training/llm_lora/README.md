# llm_lora — small-LLM LoRA

## Local, on Apple Silicon (MLX) — free, no HF login

For a tiny model (e.g. Qwen2.5-0.5B) the cheapest path is local LoRA with **MLX** (the CUDA
`train.py` below won't run on a Mac — `bitsandbytes` is CUDA-only).

```bash
uv pip install mlx-lm --python backend/.venv/bin/python          # one-time

backend/.venv/bin/python training/llm_lora/mlx_make_data.py       # -> mlx_data/{train,valid}.jsonl

backend/.venv/bin/mlx_lm.lora \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train --data training/llm_lora/mlx_data \
  --iters 800 --batch-size 8 \
  --adapter-path training/llm_lora/adapters
```
~10–25 min on Apple Silicon for the 0.5B (downloads the base ~1GB on first run; watch the loss
drop). Faster/lighter base: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`. CLI alt: `python -m mlx_lm lora`.

Then fuse + serve to eval (see "eval-ing it" below):
```bash
backend/.venv/bin/mlx_lm.fuse --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-path training/llm_lora/adapters --save-path training/llm_lora/output
```

## Cloud (HF Jobs)

A LoRA fine-tune of a small instruct LLM (e.g. `Qwen2.5-1.5B`) on the categorization task. Heavier than
the encoders (wants a GPU), so it runs on **HF Jobs** rather than locally. `train.py` is a self-contained
**uv script** (deps in its `# /// script` header).

## Prereqs
- HF **Pro** (Jobs is Pro/Team/Enterprise), and `HF_TOKEN` with write access.
- Push the dataset once so the Job can read it:
  ```python
  from data.store import to_hf_datasetdict
  to_hf_datasetdict().push_to_hub("<ns>/transclassify-data", private=True)
  ```

## Run on a GPU Job
```bash
hf jobs uv run --flavor a100-large --secrets HF_TOKEN \
  training/llm_lora/train.py \
  --dataset <ns>/transclassify-data \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --hf-repo <ns>/transclassify-qwen-lora
```
Billed per-minute; a small LoRA run is typically a few dollars. The adapter is pushed to `--hf-repo`.

## Then eval it
Serving a LoRA'd LLM for the harness is the extra step (merge → GGUF → Ollama, or load via
transformers/LiteLLM) — a follow-on, not part of v1. `AutoTrain` is a no-code alternative to this script.

> Scaffold: review the instruction prompt and hyperparameters in `train.py` before a real run.
