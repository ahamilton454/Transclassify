"""Bi-encoder categorizer: embed the transaction and each candidate category
independently, pick the nearest by cosine. Fast, cacheable, local, $0 — but it
compresses each side to one vector before they interact, so it can't reason.
"""
from .categorizer import BiEncoderCategorizer

__all__ = ["BiEncoderCategorizer"]
