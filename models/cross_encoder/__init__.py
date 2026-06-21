"""Cross-encoder categorizer: score each (transaction, category) pair jointly with
a reranker and pick the top. More accurate than the bi-encoder (token-level
interaction) but uncacheable — one model pass per candidate per row. Local, $0.
"""
from .categorizer import CrossEncoderCategorizer

__all__ = ["CrossEncoderCategorizer"]
