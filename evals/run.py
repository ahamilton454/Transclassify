"""Categorization eval runner.

Runs one categorizer (a strategy from /models + its params) against one or more
eval sets, scoring stratified accuracy, and prints a scorecard + writes a JSON report.

  backend/.venv/bin/python evals/run.py --set hand_labelled --param model=openai/gpt-5.2
  backend/.venv/bin/python evals/run.py --set all --categorizer llm_incontext --param model=gpt-5-mini

The categorizer is selected with --categorizer (which thing in /models); its
constructor args are passed generically with repeatable --param KEY=VALUE, so this
runner makes no assumption that the model is an LLM (a bi-encoder would take, e.g.,
--param checkpoint=BAAI/bge-small-en).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Make `import models` (installed editable), `import app` (backend config/.env/key
# export), and `import evals` (this package) all resolve.
for _p in (str(ROOT), str(ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.config import settings  # noqa: E402 — triggers .env load + key export
from models.llm_incontext import engine  # noqa: E402
from models.registry import get_categorizer  # noqa: E402

from data.store import load  # noqa: E402

from evals.score import ScoreReport, row_details, score  # noqa: E402

EVALS_DIR = ROOT / "evals"
# Eval "sets" are sources in the unified data layer (all on split="eval").
SETS = ["llm_generated", "dodatathings", "hand_labelled"]


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def parse_params(items: list[str]) -> dict:
    """Turn ['model=openai/gpt-5.2', 'concurrency=16'] into a kwargs dict."""
    params: dict = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--param must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        params[key.strip()] = _coerce(value.strip())
    return params


def label_for(categorizer: str, params: dict) -> str:
    inner = ", ".join(f"{k}={v}" for k, v in params.items())
    return f"{categorizer}({inner})" if inner else categorizer


async def run_set(
    set_name: str, categorizer_name: str, params: dict, limit: int | None
) -> tuple[ScoreReport, float, list[dict]]:
    records = load(split="eval", source=set_name)
    if not records:
        raise SystemExit(
            f"no eval data for source '{set_name}' under data/sources/{set_name}/."
            + (" Run data/fetch_dodatathings.py first." if set_name == "dodatathings" else "")
        )
    if limit:
        records = records[:limit]

    categorizer = get_categorizer(categorizer_name, **params)
    start = time.perf_counter()
    results = await engine.gather_bounded(
        [categorizer.categorize_one(r.transaction, r.categories) for r in records]
    )
    wall = time.perf_counter() - start
    return score(records, results), wall, row_details(records, results)


def print_scorecard(set_name: str, label: str, report: ScoreReport, wall: float) -> None:
    print(f"\n=== {set_name}  ·  {label}  ·  n={report.n} ===")
    print(f"overall accuracy : {report.overall_accuracy:.1%}")
    print(f"TAIL accuracy    : {report.tail_accuracy:.1%}   <- headline")
    if report.flip_pairs_total:
        print(f"flip-pair pass   : {report.flip_pairs_passed}/{report.flip_pairs_total}")
    if report.by_set:
        print("by taxonomy:")
        for name in sorted(report.by_set):
            s = report.by_set[name]
            print(f"  {name:<26} {s.accuracy:>6.1%}  ({s.correct}/{s.total})")
    if report.strata:
        print("by stratum:")
        for name in sorted(report.strata, key=lambda s: (s != "tail", s)):
            s = report.strata[name]
            print(f"  {name:<26} {s.accuracy:>6.1%}  ({s.correct}/{s.total})")
    if report.by_category:
        print("by category (worst first):")
        for name in sorted(report.by_category, key=lambda c: report.by_category[c].accuracy):
            c = report.by_category[name]
            print(f"  {name:<26} {c.accuracy:>6.1%}  ({c.correct}/{c.total})")
    print(f"errors: {report.errors}   cost: ${report.total_cost_usd:.5f}   wall: {wall:.1f}s")


def _trunc(text: str, n: int = 40) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def print_rows(details: list[dict], show: str) -> None:
    """show='misses' prints only failures; 'all' prints every row; 'none' prints nothing."""
    if show == "none":
        return
    rows = details if show == "all" else [d for d in details if not d["correct"]]
    if not rows:
        print("(no misses)" if show == "misses" else "(no rows)")
        return
    header = "all rows:" if show == "all" else f"misses ({len(rows)}):"
    print(header)
    for d in rows:
        mark = "✓" if d["correct"] else "✗"
        verdict = d["predicted"] if not d["error"] else f"ERROR: {_trunc(d['error'], 50)}"
        print(
            f"  {mark} {str(verdict):<16} (gold: {d['gold']:<14}) "
            f"{_trunc(d['description'])}  [{', '.join(d['strata'])}]"
        )


def write_report(
    set_name: str,
    categorizer: str,
    params: dict,
    report: ScoreReport,
    wall: float,
    details: list[dict],
    out: Path,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "set": set_name,
        "categorizer": categorizer,
        "params": params,
        "label": label_for(categorizer, params),
        "wall_seconds": round(wall, 2),
        **asdict(report),
        "overall_accuracy": report.overall_accuracy,
        "tail_accuracy": report.tail_accuracy,
        "rows": details,
    }
    out.write_text(json.dumps(payload, indent=2, default=lambda o: asdict(o)))
    print(f"wrote {out}")


async def main() -> None:
    ap = argparse.ArgumentParser(description="Run a categorization eval.")
    ap.add_argument("--set", default="llm_generated", choices=[*SETS, "all"])
    ap.add_argument(
        "--categorizer", default="llm_incontext", help="which strategy from /models (registry key)"
    )
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="categorizer constructor arg, repeatable (e.g. model=openai/gpt-5.2)",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--show",
        default="misses",
        choices=["misses", "all", "none"],
        help="per-row terminal output (full rows always saved to the JSON report)",
    )
    ap.add_argument("--out", default=None, help="JSON report path (default results/<set>.json)")
    args = ap.parse_args()

    params = parse_params(args.param)
    # Convenience: the LLM strategy needs a model id; default it from settings.
    if args.categorizer == "llm_incontext" and "model" not in params:
        params["model"] = settings.transclassify_model

    targets = SETS if args.set == "all" else [args.set]
    for set_name in targets:
        report, wall, details = await run_set(set_name, args.categorizer, params, args.limit)
        print_scorecard(set_name, label_for(args.categorizer, params), report, wall)
        print_rows(details, args.show)
        out = Path(args.out) if args.out else EVALS_DIR / "results" / f"{set_name}.json"
        write_report(set_name, args.categorizer, params, report, wall, details, out)


if __name__ == "__main__":
    asyncio.run(main())
