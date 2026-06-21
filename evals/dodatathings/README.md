# dodatathings — external real benchmark

Public, ungated dataset (`DoDataThings/us-bank-transaction-categories-v2`): ~68k
realistic, sign-prefixed US bank transaction strings across ~17 spending categories,
500+ real merchants. Good for an **external accuracy number** and **regression**.

Two limits to remember: it uses **one fixed taxonomy** (so it does *not* test
bring-your-own-categories), and the **gold is a community upload's labels** (audit a
sample; public data may also be partly memorized by frontier models). Use it for
regression, not as the trusted decision-maker — that's `hand_labelled`.

> Why not MBD / banking77 / mitulshah? MBD has no raw merchant strings (anonymized
> event codes, client-level labels); banking77 is *intent* classification, not
> categorization; mitulshah is a gated personal upload. This one matches our actual
> task: messy string → spending category.

## Prepare (fetches the real data)

```bash
backend/.venv/bin/python evals/dodatathings/prepare.py --limit 200
```
Writes `data.jsonl` (gitignored) + `evals/category_sets/dodatathings_us_v2.json`. Then:
```bash
backend/.venv/bin/python evals/run.py --set dodatathings --param model=openai/gpt-5.2
```
