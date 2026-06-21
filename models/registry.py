"""Select a categorizer strategy by name.

`name` is the strategy (e.g. "llm_incontext"); strategy-specific params (like the
LiteLLM `model` id for the LLM strategy) are passed through. This is the seam where
future strategies (embedding, cross-encoder, distilled) plug in.
"""
from __future__ import annotations

from .base import Categorizer
from .bi_encoder.categorizer import BiEncoderCategorizer
from .cross_encoder.categorizer import CrossEncoderCategorizer
from .llm_incontext.categorizer import LLMInContextCategorizer

STRATEGIES = {
    "llm_incontext": LLMInContextCategorizer,
    "bi_encoder": BiEncoderCategorizer,
    "cross_encoder": CrossEncoderCategorizer,
}


def get_categorizer(name: str = "llm_incontext", **params) -> Categorizer:
    try:
        strategy = STRATEGIES[name]
    except KeyError:
        raise ValueError(
            f"unknown categorizer strategy {name!r}; available: {sorted(STRATEGIES)}"
        ) from None
    return strategy(**params)
