# data — the unified dataset layer

One normalized dataset shared by **evaluation** (`split="eval"`) and **training**
(`split="train"`). Same shape everywhere; the two concerns differ only by the `split` tag.

## Model

- **`category_sets/<id>.json`** — the taxonomies (shared by train + eval).
- **`sources/<source>/transactions.jsonl`** — the transaction pool: `{id, description, …}` where
  `id = hash(description)`, so a string is stored once and duplicates collapse.
- **`sources/<source>/labelings.jsonl`** — `{transaction_id, category_set_id, gold, acceptable?,
  expected_other, pair_id?, strata, note?, split, source}`. One transaction → many labelings (a
  description-flip pair is one transaction with two labelings).

`store.py` joins these into `Example`s (`load(split=…, source=…)`), appends with dedup, queries the
JSONL in place with **DuckDB** (`overlap_transaction_ids`, `counts_by`), and exports a HF
`DatasetDict` for training (`to_hf_datasetdict`).

## Sources

| source | split | tracked? | how to (re)build |
|---|---|---|---|
| `hand_labelled` | eval | **yes** (curated, git-legible) | edit `sources/hand_labelled/*.jsonl` by hand |
| `synthetic_v2` | train + eval | no (regenerable) | `python data/generate.py --source synthetic_v2 --split {train\|eval} --total {10000\|2000}` |
| `dodatathings` | eval | no (fetched) | `python data/fetch_dodatathings.py --limit 200` |

`synthetic_v2` is the hybrid generator (LLM merchants → `templates.py` layered engine; see
`transaction_patterns.md`). Train covers the 6 in-distribution taxonomies; eval also covers 2 held-out
taxonomies (`restaurant_owner`, `ecommerce_seller`) the models never train on — a bring-your-own-categories test.

**Train ∩ eval = ∅** is enforced structurally: `generate.py` drops any string already in the opposite
split, and `store.overlap_transaction_ids("train","eval")` must return empty.

Run with the backend venv (has the LLM stack + duckdb): `backend/.venv/bin/python data/generate.py …`.
