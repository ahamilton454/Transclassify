"""FastAPI app: the two routes over the shared engine."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models.llm_incontext import engine
from models.registry import get_categorizer

from . import ai, db
from .config import settings
from .schemas import (
    CategorizeRequest,
    CategorizeResponse,
    EnrichRequest,
    EnrichResponse,
)
from .serp import SerperProvider, get_serp_provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(
    title="Transclassify",
    summary="Transaction categorization & enrichment API — bring your own categories.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok", "model": settings.transclassify_model}


@app.post("/v1/categorize", response_model=CategorizeResponse, tags=["categorize"])
async def categorize(req: CategorizeRequest) -> CategorizeResponse:
    """Categorize each transaction into one of the caller's categories."""
    categorizer = get_categorizer(
        "llm_incontext",
        model=settings.transclassify_model,
        max_retries=settings.llm_max_retries,
        concurrency=settings.llm_concurrency,
    )
    results = await categorizer.categorize_batch(req.transactions, req.categories)
    total = _sum_cost(results)
    db.log_request(
        route="categorize",
        model=settings.transclassify_model,
        transaction_count=len(req.transactions),
        payload={"total_cost_usd": total, "results": [r.model_dump() for r in results]},
    )
    return CategorizeResponse(
        model=settings.transclassify_model, total_cost_usd=total, results=results
    )


@app.post("/v1/enrich", response_model=EnrichResponse, tags=["enrich"])
async def enrich(req: EnrichRequest) -> EnrichResponse:
    """Categorize and add merchant metadata via a single web-search lookup per row."""
    serp = get_serp_provider()
    # A real (billable) query only runs when a SERP provider is configured.
    serp_unit_cost = (
        settings.serp_cost_per_query if isinstance(serp, SerperProvider) else 0.0
    )

    async def _one(tx):
        # Search on the raw description; the LLM cleans + reasons over the snippets.
        searched = False
        try:
            evidence = await serp.search(tx.description)
            searched = True
        except Exception:  # noqa: BLE001 — degrade to no-evidence, never fail the row
            evidence = ""
        res = await ai.enrich_one(tx, req.categories, evidence)
        if searched and serp_unit_cost:
            res.cost_usd = (res.cost_usd or 0.0) + serp_unit_cost
        return res

    results = await engine.gather_bounded(
        [_one(tx) for tx in req.transactions], concurrency=settings.llm_concurrency
    )
    total = _sum_cost(results)
    db.log_request(
        route="enrich",
        model=settings.transclassify_model,
        transaction_count=len(req.transactions),
        payload={"total_cost_usd": total, "results": [r.model_dump() for r in results]},
    )
    return EnrichResponse(
        model=settings.transclassify_model, total_cost_usd=total, results=results
    )


def _sum_cost(results) -> float:
    return round(sum(r.cost_usd or 0.0 for r in results), 6)
