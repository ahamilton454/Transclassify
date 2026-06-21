"""The normalized data model: Transaction (pool) + Labeling (relation) + CategorySet,
and the joined `Example` that eval/categorizers consume.
"""
from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

# Reuse the shared domain types — a dataset transaction IS an API transaction.
from models.types import Category, Transaction


def transaction_id(description: str) -> str:
    """Stable content id so the same string is stored once and dedupes across sources."""
    return hashlib.sha1(description.strip().encode("utf-8")).hexdigest()[:16]


class CategorySet(BaseModel):
    """A named taxonomy (may be hierarchical via category.parent)."""

    id: str
    categories: list[Category] = Field(min_length=1)

    def names(self) -> set[str]:
        return {c.name for c in self.categories}


class Labeling(BaseModel):
    """One taxonomy-specific gold for a transaction, plus split/source provenance."""

    transaction_id: str
    category_set_id: str
    gold: str
    acceptable: list[str] | None = None
    expected_other: bool = False
    pair_id: str | None = None
    strata: list[str] = Field(default_factory=list)
    note: str | None = None
    split: str = "eval"  # "train" | "eval"
    source: str = ""  # provenance, e.g. hand_labelled / llm_generated / training_v1


class Example(BaseModel):
    """A labeling joined to its transaction + resolved categories. The unit eval and
    the categorizers operate on (mirrors the old EvalRecord interface)."""

    transaction: Transaction
    categories: list[Category]
    category_set_id: str
    gold: str
    acceptable: list[str] | None = None
    expected_other: bool = False
    pair_id: str | None = None
    strata: list[str] = Field(default_factory=list)
    note: str | None = None
    split: str = "eval"
    source: str = ""

    def resolved_acceptable(self) -> set[str]:
        return set(self.acceptable) if self.acceptable else {self.gold}
