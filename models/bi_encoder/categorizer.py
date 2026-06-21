"""Bi-encoder strategy (SentenceTransformer + cosine)."""
from __future__ import annotations

from ..base import Categorizer
from ..encoder_common import category_text, pick
from ..types import Category, CategorizeResult, Transaction

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def _dot(a, b) -> float:
    # embeddings are L2-normalized, so cosine == dot product
    return sum(float(x) * float(y) for x, y in zip(a, b))


class BiEncoderCategorizer(Categorizer):
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model_name = model
        self._model = None  # lazy — no torch/download until first use
        self._cat_cache: dict[tuple[str, ...], list] = {}

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def _embed(self, texts: list[str]):
        self._ensure_loaded()
        return self._model.encode(texts, normalize_embeddings=True)

    async def categorize_one(
        self, transaction: Transaction, categories: list[Category]
    ) -> CategorizeResult:
        try:
            cat_texts = tuple(category_text(c) for c in categories)
            if cat_texts not in self._cat_cache:  # embed each taxonomy once, reuse across rows
                self._cat_cache[cat_texts] = self._embed(list(cat_texts))
            cat_emb = self._cat_cache[cat_texts]
            tx_emb = self._embed([transaction.description])[0]
            scores = [_dot(tx_emb, ce) for ce in cat_emb]
            return pick(transaction, categories, scores)
        except Exception as exc:  # noqa: BLE001 — one bad row must not fail the batch
            return CategorizeResult(
                id=transaction.id, merchant="", category=categories[0].name, confidence=0.0, error=str(exc)
            )

    async def categorize_batch(
        self, transactions: list[Transaction], categories: list[Category]
    ) -> list[CategorizeResult]:
        return [await self.categorize_one(tx, categories) for tx in transactions]
