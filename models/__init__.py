"""Swappable categorization models, shared by the backend and the eval harness.

This package is the categorization *contract* (`types`) plus pluggable strategy
implementations behind a common `Categorizer` interface (`base`), selectable by
name via `registry`. It intentionally depends on neither the backend nor the
evals — both of those consume it.
"""
