"""Dump the FastAPI OpenAPI schema to ../frontend/openapi.json (no server needed).

Run from backend/:  .venv/bin/python export_openapi.py
This is the source of truth the TypeScript client is generated from.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.main import app

OUT = Path(__file__).resolve().parent.parent / "frontend" / "openapi.json"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(app.openapi(), indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
