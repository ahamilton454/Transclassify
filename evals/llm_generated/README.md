# llm_generated — synthetic breadth (dummy for now)

Synthetic set for **diversity** the wild lacks: varied taxonomies with descriptions,
cryptic tail strings, `expected_other`, and description-flip pairs.

`data.jsonl` is currently a small **hand-written dummy** set so the harness has
something to run. `generate.py` documents the grounded approach to scale it
trustworthily — the key rules: **ground gold in a real fact** (not the LLM's opinion),
keep **generator ≠ judge ≠ subject**, and **audit a sample**. See `generate.py`.

Trust: lower than `hand_labelled` (synthetic). Use for coverage/stress, not the final
decision.
