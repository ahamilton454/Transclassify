# training — fine-tune the categorizers

Distills the LLM's labels into the small models. Trained artifacts go to **HF Hub** (or a local dir);
`models/` then serves them by id with no code change. The repo holds **code, not weights**.

## The loop

```
data/generate.py --source synthetic_v2 --split train  →  train.py  →  save / push_to_hub
        →  evals/run.py --param model=<path-or-hf-id>  →  scorecard (vs zero-shot + gpt-5-mini)
```

1. **Corpus** (in `data/`, not here — same generator as eval, just a different split). Trained on the
   6 in-distribution taxonomies; eval also covers 2 held-out taxonomies the models never see:
   ```bash
   backend/.venv/bin/python data/generate.py --source synthetic_v2 --split train --total 10000 \
     --taxonomies personal_budget freelancer_schedule_c small_business_coa \
                  student_budget rental_property nonprofit_org
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
   backend/.venv/bin/python evals/run.py --set synthetic_v2 --categorizer bi_encoder \
     --param model=training/bi_encoder/output      # or the HF id you pushed to
   ```

## `llm_lora` — cloud (HF Jobs)

A small-LLM QLoRA fine-tune that's heavier than the encoders. It's written as a **uv script** and run on
**HF Jobs** (GPU, per-minute, needs HF Pro), not locally — see `llm_lora/README.md`. `pyproject.toml`
here pins the training deps (used by HF Jobs / for reproducibility; `peft`/`trl` are only for `llm_lora`).

## Trained-model registry

Trained on `synthetic_v2` (9,943 examples, 6 taxonomies). Eval on `synthetic_v2` eval
(n=1,923, 8 taxonomies = the 6 trained **in-distribution** + 2 **held-out** the models never see).

| name | strategy | base | path | overall | in-dist (6) | held-out (2) |
|---|---|---|---|---|---|---|
| bge-bi-v2 | bi_encoder | bge-small-en-v1.5 | `training/bi_encoder/output` | **60.1%** | 63.6% | 49.4% |
| bge-cross-v2 | cross_encoder | bge-reranker-base | `training/cross_encoder/output` | 55.9% | 59.1% | 46.0% |
| qwen-lora-v2 | bare_category | Qwen2.5-0.5B | `training/llm_lora/output` → Ollama | 41.8%¹ | 45.6% | 30.2% |

¹ Qwen LoRA is **undertrained**: the macOS Metal watchdog (`Impacting Interactivity`) aborted training
near each validation step, so it completed only ~800 iters × batch 4 ≈ 3.2k samples seen (val loss 0.195)
vs the encoders' full epoch. Its number is a floor, not a fair ceiling — see `llm_lora/README.md` for the
watchdog mitigation (smaller batch / grad-checkpoint / disable mid-train validation).

**Realism cost:** moving from the old clean LLM-generated set to `synthetic_v2`'s realistic strings
dropped every model (bi 75.7→60.1, cross 68.5→55.9). **BYO-generalization cost:** every model loses
~13–15 points on the 2 held-out taxonomies — the honest bring-your-own-categories number. The bi-encoder
keeps a small lead throughout and is fastest, so it stays the local workhorse.
