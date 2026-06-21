"""Unit tests for the scorer (pure; no LLM, no network)."""
from __future__ import annotations

from models.types import Category, CategorizeResult, Transaction

from data.schema import Example
from evals.score import is_correct, row_details, score


def rec(rid, txid, gold, *, strata=None, acceptable=None, pair_id=None, cats=("A", "B")):
    return Example(
        transaction=Transaction(id=txid, description=f"tx-{txid}"),
        categories=[Category(name=c) for c in cats],
        category_set_id="test",
        gold=gold,
        acceptable=acceptable,
        strata=strata or [],
        pair_id=pair_id,
    )


def res(txid, category, *, error=None, cost=None):
    return CategorizeResult(
        id=txid, merchant="m", category=category, confidence=0.9, error=error, cost_usd=cost
    )


def test_is_correct_accept_set():
    r = rec("1", "1", "A", acceptable=["A", "B"])
    assert is_correct(r, "B") is True
    assert is_correct(r, "C") is False
    assert is_correct(r, None) is False


def test_is_correct_defaults_to_gold():
    r = rec("1", "1", "A")
    assert is_correct(r, "A") is True
    assert is_correct(r, "B") is False


def test_overall_and_stratified_accuracy():
    records = [
        rec("a", "1", "A", strata=["tail"]),
        rec("b", "2", "B", strata=["head"]),
    ]
    results = [res("1", "A"), res("2", "A")]  # first right, second wrong
    rep = score(records, results)
    assert rep.n == 2
    assert rep.correct == 1
    assert rep.overall_accuracy == 0.5
    assert rep.tail_accuracy == 1.0           # the one tail item was correct
    assert rep.strata["head"].accuracy == 0.0


def test_flip_pair_passes_only_if_both_correct():
    records = [
        rec("c", "3", "Shopping", pair_id="p1", cats=("Shopping", "Groceries")),
        rec("d", "4", "Groceries", pair_id="p1", cats=("Shopping", "Groceries")),
    ]
    # model ignores descriptions → predicts Shopping for both → pair fails
    rep = score(records, [res("3", "Shopping"), res("4", "Shopping")])
    assert rep.flip_pairs_total == 1
    assert rep.flip_pairs_passed == 0

    rep2 = score(records, [res("3", "Shopping"), res("4", "Groceries")])
    assert rep2.flip_pairs_passed == 1


def test_errored_row_not_correct_and_counted():
    records = [rec("a", "1", "A", strata=["tail"])]
    rep = score(records, [res("1", "A", error="boom")])
    assert rep.errors == 1
    assert rep.correct == 0          # errored row never scores correct even if category matches
    assert rep.tail_accuracy == 0.0


def test_per_category_accuracy():
    records = [
        rec("a", "1", "A", cats=("A", "B")),
        rec("b", "2", "A", cats=("A", "B")),
        rec("c", "3", "B", cats=("A", "B")),
    ]
    results = [res("1", "A"), res("2", "B"), res("3", "B")]  # gold A: 1/2 right; gold B: 1/1
    rep = score(records, results)
    assert rep.by_category["A"].correct == 1 and rep.by_category["A"].total == 2
    assert rep.by_category["B"].accuracy == 1.0


def test_row_details_captures_outcome():
    records = [
        rec("a", "1", "A", strata=["tail"]),
        rec("b", "2", "B"),
        rec("c", "3", "A"),
    ]
    results = [res("1", "A"), res("2", "A"), res("3", "?", error="boom")]
    rows = row_details(records, results)
    assert rows[0]["correct"] is True and rows[0]["predicted"] == "A"
    assert rows[1]["correct"] is False and rows[1]["predicted"] == "A" and rows[1]["gold"] == "B"
    # errored row: predicted nulled, error captured, not correct
    assert rows[2]["correct"] is False and rows[2]["predicted"] is None and rows[2]["error"] == "boom"


def test_cost_summed():
    records = [rec("a", "1", "A"), rec("b", "2", "B")]
    rep = score(records, [res("1", "A", cost=0.001), res("2", "B", cost=0.002)])
    assert rep.total_cost_usd == 0.003
    assert rep.overall_accuracy == 1.0
