"""Web-search lookup for enrichment, behind a thin swappable interface.

v1 ships a single Serper provider and a no-op provider (used when no key is
configured, and in tests). The cheapest-first cascade (memo -> deterministic ->
structured sources -> SERP) is a later iteration; here it's one synchronous
lookup per novel merchant.
"""
from __future__ import annotations

from typing import Protocol

import httpx

from .config import settings


class SerpProvider(Protocol):
    async def search(self, query: str) -> str:
        """Return a plain-text evidence blob (titles + snippets) for the query."""
        ...


class NullSerpProvider:
    """Returns no evidence. Used when SERP is unconfigured or in tests."""

    async def search(self, query: str) -> str:
        return ""


class SerperProvider:
    """https://serper.dev — cheap Google SERP API with a free tier."""

    ENDPOINT = "https://google.serper.dev/search"

    def __init__(self, api_key: str, timeout: float = 8.0) -> None:
        self._api_key = api_key
        self._timeout = timeout

    async def search(self, query: str) -> str:
        headers = {"X-API-KEY": self._api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": 5}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self.ENDPOINT, headers=headers, json=payload)
            resp.raise_for_status()
            return format_serper_results(resp.json())


def format_serper_results(data: dict) -> str:
    """Flatten a Serper response into a compact evidence blob for the LLM."""
    lines: list[str] = []

    kg = data.get("knowledgeGraph") or {}
    if kg:
        bits = [kg.get("title"), kg.get("type"), kg.get("website"), kg.get("description")]
        lines.append("Knowledge graph: " + " | ".join(b for b in bits if b))

    for item in (data.get("organic") or [])[:5]:
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        lines.append(f"{title} ({link}) — {snippet}".strip())

    return "\n".join(lines).strip()


def get_serp_provider() -> SerpProvider:
    if settings.serper_api_key:
        return SerperProvider(settings.serper_api_key, timeout=settings.serp_timeout_seconds)
    return NullSerpProvider()
