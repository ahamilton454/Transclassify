"""Unified dataset layer: a normalized transaction pool + labelings, shared by the
eval harness (`split="eval"`) and training (`split="train"`).

One shape, one system. Transactions are stored once (id = hash of the description);
labelings attach a taxonomy-specific gold to a transaction. The eval and training
concerns differ only by the `split` tag, so there's no second data format.
"""
