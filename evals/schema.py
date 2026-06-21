"""Eval record + category-set schema.

The unit of evaluation is a `(transaction, categories-with-descriptions, gold)`
triple. The category set is referenced by `category_set_id` (resolved from the
registry) or supplied inline as a one-off override. Reuses the shared
`models.types` so eval inputs are identical to API inputs.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from models.types import Category, Transaction


class CategorySet(BaseModel):
    """A named, reusable taxonomy (may be hierarchical via category.parent)."""

    id: str
    categories: list[Category] = Field(min_length=1)

    def names(self) -> set[str]:
        return {c.name for c in self.categories}


class EvalRecord(BaseModel):
    id: str
    transaction: Transaction
    # Exactly one source of categories: a registry reference or an inline override.
    category_set_id: str | None = None
    categories: list[Category] | None = None

    gold: str = Field(description="The correct category name for this taxonomy.")
    acceptable: list[str] | None = Field(
        default=None, description="Other defensible answers; defaults to [gold]."
    )
    expected_other: bool = Field(
        default=False, description="True when the right answer is the 'Other'/none bucket."
    )
    strata: list[str] = Field(default_factory=list, description="Difficulty/type tags.")
    pair_id: str | None = Field(
        default=None, description="Links a description-flip pair (both must be correct to pass)."
    )
    note: str | None = None

    def resolved_acceptable(self) -> set[str]:
        return set(self.acceptable) if self.acceptable else {self.gold}
