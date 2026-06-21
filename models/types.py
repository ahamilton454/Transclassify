"""The shared categorization contract: inputs and outputs.

These are the source of truth for the backend's API schema (re-exported there for
FastAPI/OpenAPI) and for the eval harness, so a transaction/category looks
identical everywhere.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """One raw transaction to categorize."""

    id: str = Field(description="Caller-supplied id, echoed back on the result.")
    description: str = Field(description="Raw transaction string, e.g. 'SQ *STARBUCKS 1234'.")
    amount: float | None = Field(default=None, description="Signed amount; negative = outflow.")
    date: str | None = Field(default=None, description="Transaction date (free-form / ISO).")
    # Lossless passthrough for anything else a bank export carries (currency, type,
    # mcc, counterparty, balance...). Preserved end-to-end; not yet fed to the model
    # — "which fields improve accuracy" is a deliberate future eval question.
    metadata: dict | None = Field(default=None, description="Arbitrary extra fields, preserved as-is.")


class Category(BaseModel):
    """One caller-defined category. The model picks from these names only.

    Categories may form a hierarchy via `parent` (a subcategory names its parent
    category). v1 treats the chosen label as flat for scoring; the hierarchy is
    surfaced to the model for context.
    """

    name: str = Field(description="Category name, returned verbatim when chosen.")
    description: str | None = Field(default=None, description="Optional hint/definition for the model.")
    parent: str | None = Field(default=None, description="Name of the parent category, if a subcategory.")
    id: str | None = Field(default=None, description="Optional stable id.")


class CategorizeResult(BaseModel):
    id: str
    merchant: str = Field(description="Cleaned, human-readable merchant name.")
    category: str = Field(description="Chosen category (always one of the supplied names).")
    confidence: float = Field(ge=0, le=1, description="Model self-reported 0–1 (soft signal).")
    cost_usd: float | None = Field(
        default=None, description="USD cost of this row (LLM + any web lookup), if known."
    )
    error: str | None = Field(default=None, description="Set if this row could not be processed.")
