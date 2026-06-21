"""Shared helpers for the embedding-based strategies (bi-encoder, cross-encoder).

Both turn a candidate's `Category` into a piece of text to embed/score, then pick
the argmax category and report a softmax-normalized confidence — so `confidence`
is in [0,1] and roughly comparable across strategies.
"""
from __future__ import annotations

import math

from .types import Category, CategorizeResult, Transaction


def category_text(category: Category) -> str:
    """Text representation of a candidate category for embedding/scoring.

    Includes the description (Label-Description Training measurably helps zero-shot
    matching) and the parent (subcategory context): "Parent > Name: description".
    """
    label = f"{category.parent} > {category.name}" if category.parent else category.name
    return f"{label}: {category.description}" if category.description else label


def softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    hi = max(scores)
    exps = [math.exp(s - hi) for s in scores]
    total = sum(exps)
    return [e / total for e in exps] if total else [1.0 / len(scores)] * len(scores)


def pick(
    transaction: Transaction, categories: list[Category], scores: list[float], cost_usd: float = 0.0
) -> CategorizeResult:
    """Argmax over candidate scores → result. confidence = softmax prob of the winner."""
    probs = softmax(scores)
    best = max(range(len(scores)), key=lambda i: scores[i])
    return CategorizeResult(
        id=transaction.id,
        merchant=transaction.description,  # encoders don't clean the merchant; eval scores category
        category=categories[best].name,
        confidence=probs[best],
        cost_usd=cost_usd,
    )
