"""The Categorizer interface every strategy implements.

A strategy maps a transaction + a caller-supplied category set to a
`CategorizeResult`. The LLM-in-context strategy is v1; embedding / cross-encoder /
distilled strategies can implement the same interface later and slot into the
registry with no change to the backend or evals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .types import Category, CategorizeResult, Transaction


class Categorizer(ABC):
    @abstractmethod
    async def categorize_one(
        self, transaction: Transaction, categories: list[Category]
    ) -> CategorizeResult:
        """Categorize a single transaction into one of `categories`."""

    @abstractmethod
    async def categorize_batch(
        self, transactions: list[Transaction], categories: list[Category]
    ) -> list[CategorizeResult]:
        """Categorize many transactions (results aligned to input order)."""
