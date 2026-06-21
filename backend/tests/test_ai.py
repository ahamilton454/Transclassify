"""Unit tests for the backend enrichment engine (LLM call mocked)."""
from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from models.llm_incontext import engine

from app import ai
from app.schemas import Category, Transaction


def test_build_enrich_model_enforces_enum_and_has_metadata():
    model = ai.build_enrich_model(["Food", "Transport"])
    fields = set(model.model_fields)
    assert {"website", "logo", "mcc", "recurring", "enrichment_confidence"} <= fields
    assert model(merchant="X", category="Food", confidence=0.5, website=None, logo=None,
                 mcc=None, recurring=None, enrichment_confidence=None).category == "Food"
    with pytest.raises(ValidationError):
        model(merchant="X", category="Nope", confidence=0.5, website=None, logo=None,
              mcc=None, recurring=None, enrichment_confidence=None)


def test_enrich_prompt_handles_empty_evidence():
    tx = Transaction(id="1", description="X")
    messages = ai.build_enrich_messages(tx, [Category(name="Food")], evidence="")
    assert "(no web results found)" in messages[-1]["content"]


def _allowed(response_model):
    return get_args(response_model.model_fields["category"].annotation)[0]


async def _fake_complete(model, messages, response_model, max_retries=2, mode=None):
    inst = response_model(
        merchant="Netflix",
        category=_allowed(response_model),
        confidence=0.9,
        website="https://netflix.com",
        logo=None,
        mcc="5814",
        recurring=True,
        enrichment_confidence=0.6,
    )
    return inst, 0.0005


async def test_enrich_one_maps_metadata(monkeypatch):
    monkeypatch.setattr(engine, "complete", _fake_complete)
    res = await ai.enrich_one(
        Transaction(id="e1", description="NETFLIX.COM"), [Category(name="Subscriptions")], "evidence"
    )
    assert res.website == "https://netflix.com"
    assert res.mcc == "5814"
    assert res.recurring is True
    assert res.enrichment_confidence == 0.6
    assert res.cost_usd == 0.0005


async def test_enrich_one_survives_error(monkeypatch):
    async def boom(model, messages, response_model, max_retries=2, mode=None):
        raise RuntimeError("provider down")

    monkeypatch.setattr(engine, "complete", boom)
    res = await ai.enrich_one(Transaction(id="x", description="?"), [Category(name="Food")], "")
    assert res.error == "provider down"
    assert res.category == "Food"
    assert res.confidence == 0.0
