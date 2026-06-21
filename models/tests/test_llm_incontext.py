"""Unit tests for the LLM-in-context categorizer (LLM call mocked, no network)."""
from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from models.llm_incontext import categorizer as cat
from models.llm_incontext import engine
from models.types import Category, Transaction


# --- strict-output model ---------------------------------------------------- #
def test_build_model_enforces_enum():
    model = cat.build_categorize_model(["Food", "Transport"])
    assert model(merchant="X", category="Food", confidence=0.5).category == "Food"
    with pytest.raises(ValidationError):
        model(merchant="X", category="Nope", confidence=0.5)


def test_model_json_schema_has_enum():
    schema = cat.build_categorize_model(["A", "B"]).model_json_schema()
    assert schema["properties"]["category"]["enum"] == ["A", "B"]


def test_empty_categories_raise():
    with pytest.raises(ValueError):
        engine.category_literal([])


# --- rendering -------------------------------------------------------------- #
def test_render_categories_flat_and_hierarchy():
    cats = [
        Category(name="Food & Drink", description="eating"),
        Category(name="Coffee", parent="Food & Drink"),
        Category(name="Transport"),
    ]
    out = engine.render_categories(cats)
    lines = out.splitlines()
    assert lines[0] == "- Food & Drink: eating"
    assert lines[1] == "  - Coffee"  # indented under its parent
    assert "- Transport" in out


def test_render_categories_orphan_not_dropped():
    cats = [Category(name="Coffee", parent="Missing")]
    assert "Coffee" in engine.render_categories(cats)


def test_render_transaction_includes_amount_date():
    tx = Transaction(id="1", description="SQ *STARBUCKS", amount=-5.0, date="2026-05-01")
    out = engine.render_transaction(tx)
    assert "SQ *STARBUCKS" in out and "amount: -5.0" in out and "date: 2026-05-01" in out


def test_prompt_includes_categories_and_transaction():
    tx = Transaction(id="1", description="SQ *STARBUCKS")
    msgs = cat.build_categorize_messages(tx, [Category(name="Food", description="meals")])
    user = msgs[-1]["content"]
    assert "Food: meals" in user and "SQ *STARBUCKS" in user


def test_clamp():
    assert engine.clamp(1.5) == 1.0
    assert engine.clamp(-0.2) == 0.0
    assert engine.clamp(0.42) == 0.42


def test_extract_cost_prefers_hidden_params():
    class FakeCompletion:
        _hidden_params = {"response_cost": 0.0012}

    assert engine.extract_cost(FakeCompletion()) == 0.0012


# --- orchestration (mock engine.complete) ----------------------------------- #
def _allowed(response_model):
    return get_args(response_model.model_fields["category"].annotation)[0]


async def _fake_complete(model, messages, response_model, max_retries=2, mode=None):
    inst = response_model(merchant="Test Merchant", category=_allowed(response_model), confidence=0.77)
    return inst, 0.0005


async def test_categorize_one_maps_and_clamps(monkeypatch):
    monkeypatch.setattr(engine, "complete", _fake_complete)
    c = cat.LLMInContextCategorizer(model="test")
    res = await c.categorize_one(
        Transaction(id="abc", description="SQ *STARBUCKS"),
        [Category(name="Food"), Category(name="Other")],
    )
    assert res.id == "abc"
    assert res.merchant == "Test Merchant"
    assert res.category == "Food"
    assert res.confidence == 0.77
    assert res.cost_usd == 0.0005
    assert res.error is None


async def test_categorize_one_survives_error(monkeypatch):
    async def boom(model, messages, response_model, max_retries=2, mode=None):
        raise RuntimeError("provider down")

    monkeypatch.setattr(engine, "complete", boom)
    c = cat.LLMInContextCategorizer(model="test")
    res = await c.categorize_one(Transaction(id="x", description="?"), [Category(name="Food")])
    assert res.error == "provider down"
    assert res.confidence == 0.0
    assert res.category == "Food"  # falls back to first category, never out-of-list


async def test_categorize_batch_preserves_order(monkeypatch):
    monkeypatch.setattr(engine, "complete", _fake_complete)
    c = cat.LLMInContextCategorizer(model="test")
    txns = [Transaction(id=str(i), description=f"tx{i}") for i in range(10)]
    results = await c.categorize_batch(txns, [Category(name="Food")])
    assert [r.id for r in results] == [str(i) for i in range(10)]


async def test_gather_bounded_preserves_order():
    async def make(i):
        return i

    out = await engine.gather_bounded([make(i) for i in range(20)], concurrency=4)
    assert out == list(range(20))
