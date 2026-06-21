"""Unit tests for the data store (temp dirs; no LLM, no network)."""
from __future__ import annotations

import json

import pytest

from data import store
from data.schema import Labeling, transaction_id
from data.store import (
    append,
    load,
    make_transaction,
    overlap_transaction_ids,
)


@pytest.fixture(autouse=True)
def _temp_data(tmp_path, monkeypatch):
    """Point the store at temp source + category-set dirs."""
    monkeypatch.setattr(store, "SOURCES_DIR", tmp_path / "sources")
    cats = tmp_path / "category_sets"
    cats.mkdir()
    (cats / "t.json").write_text(
        json.dumps({"id": "t", "categories": [{"name": "A"}, {"name": "B"}]})
    )
    monkeypatch.setattr(store, "CATEGORY_SETS_DIR", cats)
    return tmp_path


def _lab(tx, gold, split="eval", **kw):
    return Labeling(transaction_id=tx.id, category_set_id="t", gold=gold, split=split, source="s", **kw)


def test_transaction_id_stable_and_content_based():
    assert transaction_id("SQ *STARBUCKS") == transaction_id(" SQ *STARBUCKS ")  # trimmed
    assert transaction_id("a") != transaction_id("b")


def test_append_load_roundtrip_and_resolves_categories():
    tx = make_transaction("COFFEE CO")
    append("s", [tx], [_lab(tx, "A")])
    examples = load(split="eval")
    assert len(examples) == 1
    e = examples[0]
    assert e.gold == "A"
    assert {c.name for c in e.categories} == {"A", "B"}  # resolved from the registry
    assert e.transaction.description == "COFFEE CO"


def test_same_description_dedupes_to_one_transaction_many_labelings():
    tx = make_transaction("AMZN MKTP")
    append("s", [tx, tx], [_lab(tx, "A"), _lab(tx, "B")])  # flip-pair shape
    lines = (store.SOURCES_DIR / "s" / "transactions.jsonl").read_text().splitlines()
    assert len([t for t in lines if t.strip()]) == 1  # stored once
    assert len(load(split="eval")) == 2  # two labelings


def test_disjointness_query_flags_overlap():
    tx = make_transaction("ZELLE PAYMENT")
    append("s", [tx], [_lab(tx, "A", split="train"), _lab(tx, "A", split="eval")])
    assert overlap_transaction_ids("train", "eval") == [tx.id]


def test_load_validates_gold_in_category_set():
    tx = make_transaction("X")
    append("s", [tx], [_lab(tx, "NOPE")])
    with pytest.raises(ValueError):
        load(split="eval")
