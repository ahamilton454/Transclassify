#!/usr/bin/env bash
# Stop the backend and frontend dev servers (and optionally Postgres).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

kill_port() { # port, label
  local pids
  pids="$(lsof -tiTCP:"$1" -sTCP:LISTEN -nP 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "→ Stopping $2 (:$1)..."
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    echo "✓ $2 stopped"
  else
    echo "· $2 not running"
  fi
}

kill_port 5173 "Frontend"
kill_port 8000 "Backend"

if [ "${1:-}" = "--all" ]; then
  if docker info >/dev/null 2>&1; then
    echo "→ Stopping Postgres..."
    docker compose stop db >/dev/null 2>&1 || true
    echo "✓ Postgres stopped"
  fi
else
  echo "· Leaving Postgres running (use ./stop.sh --all to stop it too)."
fi
