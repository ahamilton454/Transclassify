"""Unit tests for the SERP layer."""
from __future__ import annotations

import httpx
import pytest
import respx

from app.serp import (
    NullSerpProvider,
    SerperProvider,
    format_serper_results,
)


async def test_null_provider_returns_empty():
    assert await NullSerpProvider().search("anything") == ""


def test_format_serper_results_flattens_kg_and_organic():
    data = {
        "knowledgeGraph": {
            "title": "Starbucks",
            "type": "Coffee shop",
            "website": "https://starbucks.com",
            "description": "Coffee company",
        },
        "organic": [
            {"title": "Starbucks Home", "link": "https://starbucks.com", "snippet": "Coffee"},
            {"title": "Wikipedia", "link": "https://en.wikipedia.org/Starbucks", "snippet": "Chain"},
        ],
    }
    out = format_serper_results(data)
    assert "Knowledge graph: Starbucks" in out
    assert "https://starbucks.com" in out
    assert "Wikipedia" in out


def test_format_serper_results_empty():
    assert format_serper_results({}) == ""


@respx.mock
async def test_serper_provider_posts_and_parses():
    route = respx.post("https://google.serper.dev/search").mock(
        return_value=httpx.Response(
            200,
            json={"organic": [{"title": "Netflix", "link": "https://netflix.com", "snippet": "Streaming"}]},
        )
    )
    provider = SerperProvider(api_key="test-key", timeout=2.0)
    out = await provider.search("NETFLIX.COM")
    assert route.called
    sent = route.calls.last.request
    assert sent.headers["X-API-KEY"] == "test-key"
    assert "Netflix" in out


@respx.mock
async def test_serper_provider_raises_on_http_error():
    respx.post("https://google.serper.dev/search").mock(return_value=httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        await SerperProvider(api_key="k").search("x")
