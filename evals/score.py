"""Scoring — stratified accuracy only (v1).

Headline is tail accuracy (the long tail is the product wedge). Ambiguity is
handled by accept-sets; description-flip pairs pass only if *both* members are
correct (proving the model used the differing descriptions). Calibration, macro-F1,
and hierarchy-aware partial credit are deliberately out of scope here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from data.schema import Example

# `results[i]` corresponds to `records[i]` (gather_bounded preserves order), so we
# match positionally — robust even when two examples share a transaction id (a
# description-flip pair is one transaction with two labelings).


def is_correct(record: Example, predicted: str | None) -> bool:
    return predicted is not None and predicted in record.resolved_acceptable()


def row_details(records: list[Example], results: list) -> list[dict]:
    """Per-row outcome (id, description, gold, predicted, correct, …) for logging."""
    rows = []
    for rec, res in zip(records, results):
        error = getattr(res, "error", None) if res is not None else None
        predicted = getattr(res, "category", None) if res is not None else None
        if error:
            predicted = None
        rows.append(
            {
                "id": rec.transaction.id,
                "description": rec.transaction.description,
                "gold": rec.gold,
                "acceptable": sorted(rec.resolved_acceptable()),
                "predicted": predicted,
                "correct": is_correct(rec, predicted),
                "strata": rec.strata,
                "cost_usd": getattr(res, "cost_usd", None) if res is not None else None,
                "error": error,
            }
        )
    return rows


@dataclass
class StratumScore:
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class ScoreReport:
    n: int = 0
    correct: int = 0
    strata: dict[str, StratumScore] = field(default_factory=dict)
    by_set: dict[str, StratumScore] = field(default_factory=dict)  # keyed by category_set_id
    by_category: dict[str, StratumScore] = field(default_factory=dict)  # keyed by gold category
    flip_pairs_passed: int = 0
    flip_pairs_total: int = 0
    errors: int = 0
    total_cost_usd: float = 0.0

    @property
    def overall_accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def tail_accuracy(self) -> float:
        s = self.strata.get("tail")
        return s.accuracy if s else 0.0


def score(records: list[Example], results: list) -> ScoreReport:
    """Score predictions against records (positional: results[i] ↔ records[i])."""
    report = ScoreReport(n=len(records))

    # Track per-pair correctness for the flip-pair pass rate.
    pair_members: dict[str, list[bool]] = {}

    for rec, res in zip(records, results):
        predicted = getattr(res, "category", None) if res is not None else None
        if res is not None and getattr(res, "error", None):
            report.errors += 1
            predicted = None  # an errored row is never counted correct
        if res is not None and getattr(res, "cost_usd", None):
            report.total_cost_usd += res.cost_usd

        correct = is_correct(rec, predicted)
        if correct:
            report.correct += 1

        for stratum in rec.strata:
            s = report.strata.setdefault(stratum, StratumScore())
            s.total += 1
            if correct:
                s.correct += 1

        # per-category accuracy, keyed by the gold category
        cat = report.by_category.setdefault(rec.gold, StratumScore())
        cat.total += 1
        if correct:
            cat.correct += 1

        # per-taxonomy accuracy, keyed by category_set_id
        if rec.category_set_id:
            cs = report.by_set.setdefault(rec.category_set_id, StratumScore())
            cs.total += 1
            if correct:
                cs.correct += 1

        if rec.pair_id:
            pair_members.setdefault(rec.pair_id, []).append(correct)

    for members in pair_members.values():
        report.flip_pairs_total += 1
        if all(members):
            report.flip_pairs_passed += 1

    report.total_cost_usd = round(report.total_cost_usd, 6)
    return report
