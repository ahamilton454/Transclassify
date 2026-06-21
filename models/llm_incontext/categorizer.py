"""The LLM-in-context categorizer strategy."""
from __future__ import annotations

from pydantic import BaseModel, Field, create_model

from ..base import Categorizer
from ..types import Category, CategorizeResult, Transaction
from . import engine

DEFAULT_MODEL = "openai/gpt-5.5"


def build_categorize_model(category_names: list[str]) -> type[BaseModel]:
    """Dynamic strict-output model whose `category` is an enum of the caller's names."""
    return create_model(
        "CategorizationOutput",
        merchant=(str, Field(description="Cleaned, human-readable merchant name.")),
        category=(
            engine.category_literal(category_names),
            Field(description="Single best-fit category, chosen strictly from the allowed list."),
        ),
        confidence=(float, Field(ge=0, le=1, description="Your 0–1 confidence in this categorization.")),
    )


def build_categorize_messages(tx: Transaction, categories: list[Category]) -> list[dict]:
    return [
        {"role": "system", "content": engine.CATEGORIZE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Allowed categories:\n{engine.render_categories(categories)}\n\n"
                f"Transaction:\n{engine.render_transaction(tx)}"
            ),
        },
    ]


class LLMInContextCategorizer(Categorizer):
    """Put the taxonomy in the prompt; constrain the output category to it."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_retries: int = engine.DEFAULT_MAX_RETRIES,
        concurrency: int = engine.DEFAULT_CONCURRENCY,
        mode: str | None = None,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.concurrency = concurrency
        # Small/local models (Ollama) don't do tool-calling reliably — use JSON mode.
        if mode is None and model.startswith(("ollama/", "ollama_chat/")):
            mode = "json"
        self.mode = mode

    async def categorize_one(
        self, transaction: Transaction, categories: list[Category]
    ) -> CategorizeResult:
        output_model = build_categorize_model([c.name for c in categories])
        messages = build_categorize_messages(transaction, categories)
        try:
            out, cost = await engine.complete(
                self.model, messages, output_model, self.max_retries, mode=self.mode
            )
        except Exception as exc:  # noqa: BLE001 — one bad row must not fail the batch
            return CategorizeResult(
                id=transaction.id,
                merchant="",
                category=categories[0].name,  # safe fallback; never out-of-list
                confidence=0.0,
                error=str(exc),
            )
        return CategorizeResult(
            id=transaction.id,
            merchant=out.merchant,
            category=out.category,
            confidence=engine.clamp(out.confidence),
            cost_usd=cost,
        )

    async def categorize_batch(
        self, transactions: list[Transaction], categories: list[Category]
    ) -> list[CategorizeResult]:
        return await engine.gather_bounded(
            [self.categorize_one(tx, categories) for tx in transactions],
            concurrency=self.concurrency,
        )
