"""Shared LLM primitives used by the categorizer (and reused by the backend's
enrichment path): prompt rendering, the single structured-output call, cost
extraction, and bounded concurrency.

The one network call lives in ``complete`` so everything else is pure and
unit-testable, and tests can monkeypatch ``engine.complete`` to avoid a real
provider. ``complete`` takes the model id as an argument — this module depends on
neither the backend config nor the evals.
"""
from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel

from ..types import Category, Transaction

DEFAULT_MAX_RETRIES = 2
DEFAULT_CONCURRENCY = 8

CATEGORIZE_SYSTEM = (
    "You are a precise transaction categorization engine. You receive one raw bank or "
    "card transaction string and a fixed list of allowed categories (some may be "
    "subcategories, shown indented under their parent). Clean the merchant name into "
    "something a human would recognize (strip processor prefixes like 'SQ *', store "
    "numbers, and noise). Choose the single best-fit category strictly from the allowed "
    "list, honoring each category's description when given — the user's definition wins "
    "over your own assumptions. Report calibrated confidence between 0 and 1; use low "
    "confidence when the string is ambiguous or the merchant is unknown."
)


# --------------------------------------------------------------------------- #
# Strict-output enum + rendering
# --------------------------------------------------------------------------- #
def category_literal(category_names: list[str]):
    """Literal[*names] — built dynamically so the model can only return an in-list value."""
    if not category_names:
        raise ValueError("at least one category is required")
    return Literal[tuple(category_names)]  # type: ignore[valid-type]


def render_categories(categories: list[Category]) -> str:
    """Render the taxonomy as an indented tree so the model sees subcategory structure."""
    children: dict[str | None, list[Category]] = {}
    for c in categories:
        children.setdefault(c.parent, []).append(c)
    names = {c.name for c in categories}
    lines: list[str] = []

    def emit(cat: Category, depth: int) -> None:
        indent = "  " * depth
        line = f"{indent}- {cat.name}" + (f": {cat.description}" if cat.description else "")
        lines.append(line)
        for child in children.get(cat.name, []):
            emit(child, depth + 1)

    for top in children.get(None, []):
        emit(top, 0)
    # Orphans (parent named but not present in the set) — render flat so nothing is dropped.
    for c in categories:
        if c.parent is not None and c.parent not in names:
            lines.append(f"- {c.name}" + (f": {c.description}" if c.description else ""))
    return "\n".join(lines)


def render_transaction(tx: Transaction) -> str:
    parts = [f"description: {tx.description!r}"]
    if tx.amount is not None:
        parts.append(f"amount: {tx.amount}")
    if tx.date is not None:
        parts.append(f"date: {tx.date}")
    return "\n".join(parts)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# --------------------------------------------------------------------------- #
# The single LLM call (monkeypatch this in tests)
# --------------------------------------------------------------------------- #
async def complete(
    model: str,
    messages: list[dict],
    response_model: type[BaseModel],
    max_retries: int = DEFAULT_MAX_RETRIES,
    mode: str | None = None,
) -> tuple[BaseModel, float | None]:
    """Run one structured-output completion via Instructor + LiteLLM.

    Returns (parsed_output, cost_usd). cost_usd is None if it can't be determined.
    `mode` selects the Instructor structured-output mode (e.g. "json" for small/local
    models that don't do tool-calling well); None uses Instructor's default (tools).
    Imported lazily so this module loads without the heavy LLM stack or any key.
    """
    import instructor
    from litellm import acompletion

    if mode:
        client = instructor.from_litellm(acompletion, mode=getattr(instructor.Mode, mode.upper()))
    else:
        client = instructor.from_litellm(acompletion)
    parsed, raw = await client.chat.completions.create_with_completion(
        model=model,
        response_model=response_model,
        messages=messages,
        max_retries=max_retries,
    )
    return parsed, extract_cost(raw)


def extract_cost(completion) -> float | None:
    """Pull the USD cost LiteLLM attaches to the raw response."""
    hidden = getattr(completion, "_hidden_params", None) or {}
    cost = hidden.get("response_cost")
    if cost is not None:
        return float(cost)
    try:  # fall back to recomputing from the response
        import litellm

        return float(litellm.completion_cost(completion_response=completion))
    except Exception:  # noqa: BLE001 — cost is best-effort, never fatal
        return None


async def gather_bounded(coros: list, concurrency: int = DEFAULT_CONCURRENCY) -> list:
    """Run coroutines with bounded concurrency (protects providers + sockets)."""
    sem = asyncio.Semaphore(concurrency)

    async def _run(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*[_run(c) for c in coros])
