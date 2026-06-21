"""Unit tests for the bi-/cross-encoder strategies with the model mocked (no torch)."""
from __future__ import annotations

from models.bi_encoder.categorizer import BiEncoderCategorizer
from models.cross_encoder.categorizer import CrossEncoderCategorizer
from models.encoder_common import category_text, pick, softmax
from models.registry import STRATEGIES
from models.types import Category, Transaction

COFFEE = Category(name="Coffee")
TRANSPORT = Category(name="Transport")
TX = Transaction(id="1", description="SBUX COFFEE 4471")


# --- encoder_common --------------------------------------------------------- #
def test_category_text_renders_parent_and_description():
    c = Category(name="Coffee", parent="Food & Drink", description="cafes")
    assert category_text(c) == "Food & Drink > Coffee: cafes"
    assert category_text(Category(name="Transport")) == "Transport"


def test_softmax_sums_to_one():
    p = softmax([2.0, 1.0, 0.0])
    assert abs(sum(p) - 1.0) < 1e-9
    assert p[0] > p[1] > p[2]


def test_pick_argmax_and_confidence_range():
    res = pick(TX, [COFFEE, TRANSPORT], scores=[0.9, 0.1])
    assert res.category == "Coffee"
    assert 0.0 <= res.confidence <= 1.0
    assert res.merchant == TX.description  # encoders pass the raw string through
    assert res.cost_usd == 0.0


# --- fakes (deterministic, keyword-driven) ---------------------------------- #
class FakeST:
    """Returns a 2D vector: 'coffee' texts -> [1,0], others -> [0,1]."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True):
        self.calls += 1
        return [[1.0, 0.0] if "coffee" in t.lower() else [0.0, 1.0] for t in texts]


class FakeCE:
    def predict(self, pairs):
        return [1.0 if "coffee" in q.lower() and "coffee" in p.lower() else 0.0 for q, p in pairs]


class Raises:
    def encode(self, *a, **k):
        raise RuntimeError("boom")

    def predict(self, *a, **k):
        raise RuntimeError("boom")


# --- bi-encoder ------------------------------------------------------------- #
async def test_bi_encoder_selects_and_caches_categories():
    c = BiEncoderCategorizer()
    c._model = FakeST()  # inject; bypasses lazy load
    r1 = await c.categorize_one(TX, [COFFEE, TRANSPORT])
    assert r1.category == "Coffee"
    assert 0.0 <= r1.confidence <= 1.0
    await c.categorize_one(TX, [COFFEE, TRANSPORT])
    # category set embedded once (cached), transaction embedded each call → 3 total
    assert c._model.calls == 3


async def test_bi_encoder_error_falls_back():
    c = BiEncoderCategorizer()
    c._model = Raises()
    r = await c.categorize_one(TX, [COFFEE, TRANSPORT])
    assert r.error and "boom" in r.error
    assert r.category == "Coffee"  # first category, never out-of-list
    assert r.confidence == 0.0


# --- cross-encoder ---------------------------------------------------------- #
async def test_cross_encoder_selects():
    c = CrossEncoderCategorizer()
    c._model = FakeCE()
    r = await c.categorize_one(TX, [COFFEE, TRANSPORT])
    assert r.category == "Coffee"
    assert 0.0 <= r.confidence <= 1.0


async def test_cross_encoder_error_falls_back():
    c = CrossEncoderCategorizer()
    c._model = Raises()
    r = await c.categorize_one(TX, [COFFEE, TRANSPORT])
    assert r.error and "boom" in r.error
    assert r.category == "Coffee"


def test_registry_has_encoders():
    assert "bi_encoder" in STRATEGIES and "cross_encoder" in STRATEGIES
