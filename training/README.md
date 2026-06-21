# training — fine-tune the categorizers

Distills the LLM's labels into the small models. Trained artifacts go to **HF Hub** (or a local dir);
`models/` then serves them by id with no code change. The repo holds **code, not weights**.

## The loop

```
data/generate.py --source training_v1 --split train   →  train.py  →  save / push_to_hub
        →  evals/run.py --param model=<path-or-hf-id>  →  scorecard (vs zero-shot + gpt-5-mini)
```

1. **Corpus** (in `data/`, not here — same generator as eval, just a different split):
   ```bash
   backend/.venv/bin/python data/generate.py --source training_v1 --split train --total 5000
   ```
   Disjoint from eval by construction (`store.overlap_transaction_ids("train","eval")` must be empty).

2. **Train** (encoders run locally with the **backend venv** — it already has sentence-transformers;
   no separate venv needed):
   ```bash
   backend/.venv/bin/python training/bi_encoder/train.py   [--hf-repo <ns>/<name>]
   backend/.venv/bin/python training/cross_encoder/train.py [--hf-repo <ns>/<name>]
   ```

3. **Eval** the trained model through the existing harness:
   ```bash
   backend/.venv/bin/python evals/run.py --set llm_generated --categorizer bi_encoder \
     --param model=training/bi_encoder/output      # or the HF id you pushed to
   ```

## `llm_lora` — cloud (HF Jobs)

A small-LLM QLoRA fine-tune that's heavier than the encoders. It's written as a **uv script** and run on
**HF Jobs** (GPU, per-minute, needs HF Pro), not locally — see `llm_lora/README.md`. `pyproject.toml`
here pins the training deps (used by HF Jobs / for reproducibility; `peft`/`trl` are only for `llm_lora`).

## Trained-model registry

| name | strategy | base | HF id / path | eval (llm_generated overall) |
|---|---|---|---|---|
| bge-bi-v1 | bi_encoder | bge-small-en-v1.5 | `training/bi_encoder/output` | **75.7%** (zero-shot was 50.0%) |
| _bge-cross-v1_ | cross_encoder | bge-reranker-base | _(fill)_ | _(fill)_ |

Reference points on `llm_generated` (n=1013): zero-shot bi 50% · zero-shot cross 38% · gpt-5-mini 86%.
The bi-encoder fine-tune (1 epoch, 5.8k synthetic examples, ~44s on a Mac) jumped **50 → 75.7%**.
