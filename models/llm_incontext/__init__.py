"""LLM-in-context categorizer: put the caller's taxonomy in the prompt and let a
frontier model reason over the messy transaction string, with the chosen category
constrained (strict structured output) to the supplied names.
"""
from .categorizer import LLMInContextCategorizer

__all__ = ["LLMInContextCategorizer"]
