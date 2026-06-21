# Transclassify

Transaction **categorization & enrichment** API — bring your own categories.

Incumbents classify into a fixed taxonomy and lean on lookup tables. Transclassify lets the
caller supply their own categories per request and has an LLM reason over the messy transaction
string in-context — winning on the long tail (cryptic processor strings, novel merchants) where
matching fails.

This is the **v1**: two routes, a CSV-upload web client, one model for both routes, no caching,
no auth. (Caching/merchant-memo, the SERP cascade, auth, and billing are deliberately deferred.)

## Layout

```
backend/    FastAPI (Python) — the engine + /v1/categorize and /v1/enrich
frontend/   Vite + React + TanStack (Query/Table) — CSV upload → results grid
docker-compose.yml   Postgres (optional; request logging only)
```

## Routes

- `POST /v1/categorize` → `{ merchant, category, confidence }` per transaction. `category` is
  constrained (OpenAI Structured Outputs `enum`) to the caller's category names. `confidence` is
  the model's self-reported 0–1 (a soft signal, not calibrated).
- `POST /v1/enrich` → the above **plus** `{ website, logo, mcc, recurring, enrichment_confidence }`,
  using a single Serper web lookup per row fed to the same model. SERP sits behind a swappable
  `SerpProvider`; with no `SERPER_API_KEY` it degrades to no-evidence (still categorizes).

## Quickstart

### 1. Backend

```bash
cp .env.example .env          # add OPENAI_API_KEY (and optionally SERPER_API_KEY)
docker compose up -d          # optional: Postgres for request logging
cd backend
uv venv && uv pip install -e '.[dev]'
.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the OpenAPI UI.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev                   # predev regenerates the typed client from the backend schema
```

Open http://localhost:5173 — drop `sample-data/transactions.csv`, set categories, run.
(The dev server proxies `/v1` and `/health` to the backend on :8000.)

## The type bridge (Python → TypeScript)

Pydantic models → FastAPI `/openapi.json` → `@hey-api/openapi-ts` → typed TS SDK. After changing a
backend schema, regenerate:

```bash
cd backend && .venv/bin/python export_openapi.py     # writes frontend/openapi.json
cd ../frontend && npm run gen:api                     # regenerates src/api/generated (also a predev hook)
```

Mismatches between backend and frontend then surface at compile time (`tsc`).

## Model selection

Default `openai/gpt-5.5` (quality-first), set via `TRANSCLASSIFY_MODEL`. Swap to a cheaper model
(`openai/gpt-5.4-mini`, `gemini/gemini-2.5-flash`, etc.) by changing that one env var — the AI layer
is LiteLLM-backed. Categorization token cost is negligible; the cost lever is the (deferred) SERP
cascade for enrichment.

## Tests

```bash
cd backend && .venv/bin/python -m pytest        # engine, SERP, routes (LLM mocked, no network)
cd frontend && npm run test                     # CSV parsing/column-detection, category parsing
```

Copy uses "categorize", never "classify"; no accuracy claims until a benchmark backs them.
