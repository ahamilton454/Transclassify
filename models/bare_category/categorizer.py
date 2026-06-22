"""The bare-category strategy: for small fine-tuned LLMs that emit ONLY a category.

Unlike `llm_incontext` (which asks for a strict `{merchant, category, confidence}`
JSON object), this talks to the model the way our LoRA was *trained* (see
`training/llm_lora/mlx_make_data.py`): the training prompt in, raw text out, then
the text is normalized back onto the allowed category names (case-insensitively).
No Instructor, no enum rejection, no retries — so the fine-tune is measured fairly.

Pair it with an Ollama-served model:
    --categorizer bare_category --param model=ollama_chat/transclassify-qwen
"""
from __future__ import annotations

from ..base import Categorizer
from ..llm_incontext import engine
from ..types import Category, CategorizeResult, Transaction


def render_categories(categories: list[Category]) -> str:
    """Match training/llm_lora/mlx_make_data.py exactly so inference == training."""
    lines = []
    for c in categories:
        label = f"{c.parent} > {c.name}" if c.parent else c.name
        lines.append(f"- {label}" + (f": {c.description}" if c.description else ""))
    return "\n".join(lines)


def build_prompt(tx: Transaction, categories: list[Category]) -> str:
    return (
        "Categorize the transaction into exactly one of the allowed categories. "
        "Respond with only the category name.\n\n"
        f"Allowed categories:\n{render_categories(categories)}\n\n"
        f"Transaction: {tx.description}"
    )


def normalize(raw: str, categories: list[Category]) -> tuple[str, bool]:
    """Map the model's free text onto an allowed name. Returns (category, matched)."""
    text = (raw or "").strip().strip("\"'`").strip()
    first_line = text.splitlines()[0].strip() if text else ""
    by_lower = {c.name.lower(): c.name for c in categories}
    for candidate in (text, first_line):
        hit = by_lower.get(candidate.lower())
        if hit is not None:
            return hit, True
    # Unmatched → "Other" if the taxonomy has it, else the first category.
    fallback = by_lower.get("other", categories[0].name)
    return fallback, False


class BareCategoryCategorizer(Categorizer):
    """Query a category-only fine-tuned LLM via LiteLLM and normalize its output."""

    def __init__(self, model: str, concurrency: int = engine.DEFAULT_CONCURRENCY) -> None:
        self.model = model
        self.concurrency = concurrency

    async def categorize_one(
        self, transaction: Transaction, categories: list[Category]
    ) -> CategorizeResult:
        from litellm import acompletion

        prompt = build_prompt(transaction, categories)
        try:
            resp = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 — one bad row must not fail the batch
            return CategorizeResult(
                id=transaction.id,
                merchant="",
                category=categories[0].name,
                confidence=0.0,
                error=str(exc),
            )
        category, matched = normalize(raw, categories)
        return CategorizeResult(
            id=transaction.id,
            merchant="",  # this model only emits a category
            category=category,
            confidence=1.0 if matched else 0.0,
            cost_usd=engine.extract_cost(resp),
        )

    async def categorize_batch(
        self, transactions: list[Transaction], categories: list[Category]
    ) -> list[CategorizeResult]:
        return await engine.gather_bounded(
            [self.categorize_one(tx, categories) for tx in transactions],
            concurrency=self.concurrency,
        )
