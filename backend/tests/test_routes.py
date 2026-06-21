"""Route-level tests with the LLM call mocked (no network, no API key)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from models.llm_incontext import engine

from app.main import app

client = TestClient(app)


# Both routes (categorize via the registry categorizer, enrich via app.ai) flow
# through engine.complete, so one mock covers everything.
async def _fake_complete(model, messages, response_model, max_retries=2, mode=None):
    enum = response_model.model_json_schema()["properties"]["category"]["enum"]
    kwargs = dict(merchant="Starbucks", category=enum[0], confidence=0.91)
    if "website" in response_model.model_fields:
        kwargs.update(
            website="https://starbucks.com",
            logo=None,
            mcc="5814",
            recurring=True,
            enrichment_confidence=0.7,
        )
    return response_model(**kwargs), 0.0005


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setattr(engine, "complete", _fake_complete)


CATS = [{"name": "Food & Drink"}, {"name": "Transport"}, {"name": "Income"}]


def test_categorize_route():
    body = {
        "transactions": [
            {"id": "1", "description": "SQ *STARBUCKS 1234", "amount": -5.75},
            {"id": "2", "description": "UBER *EATS 8829", "amount": -31.07},
        ],
        "categories": CATS,
    }
    resp = client.post("/v1/categorize", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["model"]
    assert len(data["results"]) == 2
    assert {r["id"] for r in data["results"]} == {"1", "2"}
    for r in data["results"]:
        assert r["category"] in {c["name"] for c in CATS}
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["cost_usd"] == 0.0005
    # total cost is the sum across rows
    assert data["total_cost_usd"] == 0.001


def test_enrich_route_includes_metadata():
    body = {
        "transactions": [{"id": "a", "description": "NETFLIX.COM", "amount": -15.49}],
        "categories": CATS,
    }
    resp = client.post("/v1/enrich", json=body)
    assert resp.status_code == 200, resp.text
    r = resp.json()["results"][0]
    assert r["website"] == "https://starbucks.com"
    assert r["mcc"] == "5814"
    assert r["recurring"] is True
    assert r["enrichment_confidence"] == 0.7


def test_categorize_validation_requires_categories():
    resp = client.post(
        "/v1/categorize",
        json={"transactions": [{"id": "1", "description": "X"}], "categories": []},
    )
    assert resp.status_code == 422


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
