#!/usr/bin/env bash
# Start Postgres, the backend, and the frontend — skipping whatever is already up.
# Idempotent: safe to run repeatedly. Logs land in .run/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LOG_DIR="$ROOT/.run"
mkdir -p "$LOG_DIR"

BACKEND_PORT=8000
FRONTEND_PORT=5173

port_in_use() { lsof -iTCP:"$1" -sTCP:LISTEN -nP >/dev/null 2>&1; }

wait_for() { # url, label
  for _ in $(seq 1 30); do
    curl -sf "$1" >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  return 1
}

echo "Transclassify — starting services"
echo "---------------------------------"

# --- Postgres (optional; logging only) -------------------------------------
if docker info >/dev/null 2>&1; then
  # `up -d` is a no-op if the container is already running.
  echo "→ Postgres: ensuring container is up..."
  docker compose up -d db >/dev/null
  echo "✓ Postgres ready on :5432"
else
  echo "⚠  Docker not running — skipping Postgres (the API works without it)."
fi

# --- Backend ---------------------------------------------------------------
if port_in_use "$BACKEND_PORT"; then
  echo "✓ Backend already running on :$BACKEND_PORT"
else
  if [ ! -d backend/.venv ]; then
    echo "→ Backend: creating venv + installing deps (first run)..."
    (cd backend && uv venv >/dev/null && uv pip install -e '.[dev]' >/dev/null)
    # Shared categorization models package (import models) — used by backend + evals.
    uv pip install -e . --python backend/.venv/bin/python >/dev/null
  fi
  if [ ! -f backend/.env ]; then
    cp .env.example backend/.env
    echo "⚠  Created backend/.env from .env.example — add your OPENAI_API_KEY to it."
  fi
  echo "→ Backend: starting on :$BACKEND_PORT..."
  (cd backend && nohup .venv/bin/python -m uvicorn app.main:app --port "$BACKEND_PORT" \
    >"$LOG_DIR/backend.log" 2>&1 &)
  if wait_for "http://localhost:$BACKEND_PORT/health"; then
    echo "✓ Backend ready on :$BACKEND_PORT"
  else
    echo "✗ Backend did not become healthy — see $LOG_DIR/backend.log"
  fi
fi

# --- Frontend --------------------------------------------------------------
if port_in_use "$FRONTEND_PORT"; then
  echo "✓ Frontend already running on :$FRONTEND_PORT"
else
  if [ ! -d frontend/node_modules ]; then
    echo "→ Frontend: installing deps (first run)..."
    (cd frontend && npm install >/dev/null 2>&1)
  fi
  echo "→ Frontend: starting on :$FRONTEND_PORT..."
  (cd frontend && nohup npm run dev >"$LOG_DIR/frontend.log" 2>&1 &)
  if wait_for "http://localhost:$FRONTEND_PORT"; then
    echo "✓ Frontend ready on :$FRONTEND_PORT"
  else
    echo "✗ Frontend did not come up — see $LOG_DIR/frontend.log"
  fi
fi

echo "---------------------------------"
echo "App:     http://localhost:$FRONTEND_PORT"
echo "API docs: http://localhost:$BACKEND_PORT/docs"
echo "Logs:    tail -f .run/backend.log  |  tail -f .run/frontend.log"
echo "Stop:    ./stop.sh"
