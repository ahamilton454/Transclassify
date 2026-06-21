"""API request/response schemas.

The core domain types (`Transaction`, `Category`, `CategorizeResult`) live in the
shared `models` package and are re-exported here so FastAPI/OpenAPI (and the
generated TS client) use the exact same contract the categorizers do.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# Re-exported shared contract — the source of truth for the API + generated types.
from models.types import Category, CategorizeResult, Transaction  # noqa: F401


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class CategorizeRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1)
    categories: list[Category] = Field(min_length=1)


class EnrichRequest(CategorizeRequest):
    """Same inputs as categorize; the route adds merchant metadata."""


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
class EnrichResult(CategorizeResult):
    website: str | None = Field(default=None)
    logo: str | None = Field(default=None, description="Logo URL if found.")
    mcc: str | None = Field(default=None, description="Merchant category code if known.")
    recurring: bool | None = Field(default=None, description="Looks like a recurring charge.")
    enrichment_confidence: float | None = Field(
        default=None, ge=0, le=1, description="Confidence in the metadata above."
    )


class CategorizeResponse(BaseModel):
    model: str
    total_cost_usd: float = Field(description="Summed USD cost across all rows.")
    results: list[CategorizeResult]


class EnrichResponse(BaseModel):
    model: str
    total_cost_usd: float = Field(description="Summed USD cost across all rows.")
    results: list[EnrichResult]
