# Categorization evals

Measures "good categorization" before we pick/tune a model. Runs the shared
`models` categorizers (exactly what the backend ships) over the **`split="eval"`** rows of the shared
`data/` layer and reports **stratified accuracy**, with **tail accuracy as the headline**.

The data itself (transactions, labelings, taxonomies) lives in **`data/`** — `evals/` is just the
harness (runner + scorer). `--set` selects a `data/` source (all on the eval split).

## Run

```bash
backend/.venv/bin/python evals/run.py --set hand_labelled --param model=openai/gpt-5.2
backend/.venv/bin/python evals/run.py --set all --param model=gpt-5-mini --limit 50
```
- `--set` — `hand_labelled` | `llm_generated` | `dodatathings` | `all`
- `--categorizer` — which strategy from `/models` (default `llm_incontext`)
- `--param KEY=VALUE` — that strategy's constructor args, repeatable. For `llm_incontext` the key is
  `model` (a LiteLLM id); a bi-encoder strategy would take e.g. `--param checkpoint=BAAI/bge-small-en`.
  Not all `/models` strategies are LLMs, so there's no `--llm` flag.
- needs a working model key in `backend/.env` for LLM strategies

## Concepts

- **Unit:** `(transaction, categories-with-descriptions, gold)` triple. The category set
  **varies** across items (tests bring-your-own-categories) and category **descriptions can
  flip the gold**.
- **Category sets** are a registry: `data/category_sets/<id>.json` (referenced by `category_set_id`).
  They support **subcategories** via `parent`. (Build/refresh sources: see `data/README.md`.)
- **Strata** tag each item (`tail`, `head`, `processor-prefix`, `p2p`, `income`, `ambiguous`,
  `description_dependent`, …); accuracy is reported per stratum.
- **accept-sets** (`acceptable`) handle ambiguity; **`expected_other`** marks "none fits".
- **Description-flip pairs** (`pair_id`) pass only if *both* members are correct — proof the model
  used the differing descriptions.

## The three sets (different trust levels)

| Set | What it tests | Trust | Role |
|---|---|---|---|
| `dodatathings` | head accuracy on a real public dataset, fixed taxonomy | medium (not our labels; leakage risk) | external benchmark / regression |
| `llm_generated` | breadth, BYO, edge cases | lower (synthetic) | stress; small **dummy** set today |
| `hand_labelled` | tail + BYO on real strings | **high** | **the decision-maker**; small, grow it |

Scoring is accuracy-only for v1; calibration, macro-F1, and hierarchy-aware partial credit are
deferred.
