# llm_lora — small-LLM QLoRA on HF Jobs

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
