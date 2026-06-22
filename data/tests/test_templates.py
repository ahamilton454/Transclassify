"""Unit tests for the template engine (deterministic; no LLM, no network)."""
from __future__ import annotations

import random

import pytest

from data import templates


def _rng():
    return random.Random(0)


def test_all_channels_produce_nonempty_single_line():
    for ch in templates.CHANNELS:
        s = templates.compose("Blue Bottle Coffee", ch, _rng())
        assert s and "\n" not in s
        assert s == s.strip()
        assert "  " not in s  # whitespace collapsed


def test_determinism_same_seed_same_string():
    a = templates.compose("Trader Joe's", "in_store", random.Random(42))
    b = templates.compose("Trader Joe's", "in_store", random.Random(42))
    assert a == b


def test_seed_variation_produces_variety():
    seen = {templates.compose("Shell", "gas", random.Random(i)) for i in range(30)}
    assert len(seen) > 5  # the engine isn't collapsing to one string


@pytest.mark.parametrize("cap", [20, 22])
def test_merchant_field_respects_cap(cap):
    out = templates.merchant_field("Some Extremely Long Merchant Name LLC", _rng(), cap=cap)
    assert len(out) <= cap
    assert out == out.upper().strip()


def test_channel_markers_present():
    # gas surfaces a pump/store form across seeds
    gas = " ".join(templates.compose("Chevron", "gas", random.Random(i)) for i in range(40))
    assert "PUMP#" in gas or "#" in gas
    # p2p always names a known app
    for i in range(20):
        p = templates.compose("Jane Doe", "p2p", random.Random(i))
        assert any(k in p for k in ("VENMO", "ZELLE", "CASH APP", "PAYPAL", "PP*"))
    # atm
    assert "ATM WITHDRAWAL" in templates.compose("", "atm_cash", random.Random(3))
    # ach biller: the biller name always surfaces; labeled ACH markers show in aggregate
    ach_lines = [templates.compose("Pacific Power", "ach_biller", random.Random(i)) for i in range(40)]
    for a in ach_lines:
        assert "PACIFIC" in a.upper()
    joined = " ".join(ach_lines)
    assert any(k in joined for k in ("DES:", "ORIG CO NAME:", "EFT DEBIT", "ACH DEBIT",
                                     "PREAUTHORIZED DEBIT", "ACH External"))


def test_amount_sign_by_channel():
    assert templates.amount_for("payroll", _rng()) > 0
    assert templates.amount_for("deposit", _rng()) > 0
    assert templates.amount_for("in_store", _rng()) < 0
    assert templates.amount_for("bank_fee", _rng()) < 0
