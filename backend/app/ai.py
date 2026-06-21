"""Enrichment engine (categorize + merchant metadata).

Categorization itself now lives in the shared `models` package; enrichment layers
SERP-backed metadata on top, reusing the same low-level LLM primitives
(`models.llm_incontext.engine`) so there's one structured-output call path.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, create_model

from models.llm_incontext import engine

from .config import settings
from .schemas import Category, EnrichResult, Transaction

ENRICH_SYSTEM = (
    engine.CATEGORIZE_SYSTEM
    + " You are also given web search results about the merchant. Use them only as "
    "supporting evidence to fill website, logo, MCC, and whether the charge is recurring. "
    "If the evidence is weak or absent, return null for that field rather than guessing, "
    "and lower enrichment_confidence accordingly."
)


def build_enrich_model(category_names: list[str]) -> type[BaseModel]:
    return create_model(
        "EnrichmentOutput",
        merchant=(str, Field(description="Cleaned, human-readable merchant name.")),
        category=(
            engine.category_literal(category_names),
            Field(description="Single best-fit category, chosen strictly from the allowed list."),
        ),
        confidence=(float, Field(ge=0, le=1, description="Your 0–1 confidence in the categorization.")),
        website=(str | None, Field(description="Official merchant website URL, or null if unknown.")),
        logo=(str | None, Field(description="Merchant logo image URL, or null if unknown.")),
        mcc=(str | None, Field(description="Merchant Category Code if identifiable, else null.")),
        recurring=(bool | None, Field(description="True if this looks like a recurring charge, else null.")),
        enrichment_confidence=(
            float | None,
            Field(description="Your 0–1 confidence in the metadata fields, or null."),
        ),
    )


def build_enrich_messages(
    tx: Transaction, categories: list[Category], evidence: str
) -> list[dict]:
    evidence_block = evidence.strip() or "(no web results found)"
    return [
        {"role": "system", "content": ENRICH_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Allowed categories:\n{engine.render_categories(categories)}\n\n"
                f"Transaction:\n{engine.render_transaction(tx)}\n\n"
                f"Web search results:\n{evidence_block}"
            ),
        },
    ]


async def enrich_one(
    tx: Transaction, categories: list[Category], evidence: str
) -> EnrichResult:
    output_model = build_enrich_model([c.name for c in categories])
    messages = build_enrich_messages(tx, categories, evidence)
    try:
        out, cost = await engine.complete(
            settings.transclassify_model, messages, output_model, settings.llm_max_retries
        )
    except Exception as exc:  # noqa: BLE001 — one bad row must not fail the batch
        return EnrichResult(
            id=tx.id, merchant="", category=categories[0].name, confidence=0.0, error=str(exc)
        )
    return EnrichResult(
        id=tx.id,
        merchant=out.merchant,
        category=out.category,
        confidence=engine.clamp(out.confidence),
        cost_usd=cost,
        website=out.website,
        logo=out.logo,
        mcc=out.mcc,
        recurring=out.recurring,
        enrichment_confidence=engine.clamp(out.enrichment_confidence)
        if out.enrichment_confidence is not None
        else None,
    )
