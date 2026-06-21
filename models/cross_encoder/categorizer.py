"""Cross-encoder strategy (CrossEncoder reranker over (transaction, category) pairs)."""
from __future__ import annotations

from ..base import Categorizer
from ..encoder_common import category_text, pick
from ..types import Category, CategorizeResult, Transaction

DEFAULT_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderCategorizer(Categorizer):
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model_name = model
        self._model = None  # lazy — no torch/download until first use

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    async def categorize_one(
        self, transaction: Transaction, categories: list[Category]
    ) -> CategorizeResult:
        try:
            self._ensure_loaded()
            pairs = [(transaction.description, category_text(c)) for c in categories]
            scores = [float(s) for s in self._model.predict(pairs)]
            return pick(transaction, categories, scores)
        except Exception as exc:  # noqa: BLE001 — one bad row must not fail the batch
            return CategorizeResult(
                id=transaction.id, merchant="", category=categories[0].name, confidence=0.0, error=str(exc)
            )

    async def categorize_batch(
        self, transactions: list[Transaction], categories: list[Category]
    ) -> list[CategorizeResult]:
        return [await self.categorize_one(tx, categories) for tx in transactions]
