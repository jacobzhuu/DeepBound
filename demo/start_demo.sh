#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="$ROOT_DIR/demo"

PYTHON_BIN="${PYTHON_BIN:-python}"
HOST_ADDR="${HOST_ADDR:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
BACKEND_LOG="$DEMO_DIR/backend.log"

check_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

detect_ip() {
  if [[ -n "${PUBLIC_IP:-}" ]]; then
    echo "$PUBLIC_IP"
    return
  fi
  if command -v ip >/dev/null 2>&1; then
    ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}'
    return
  fi
  if command -v hostname >/dev/null 2>&1; then
    hostname -I 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i !~ /^127\./) {print $i; exit}}'
    return
  fi
  echo "127.0.0.1"
}

check_cmd "$PYTHON_BIN"
check_cmd npm
check_cmd node

if [[ ! -d "$DEMO_DIR" ]]; then
  echo "Missing demo directory: $DEMO_DIR" >&2
  exit 1
fi

if [[ ! -f "$DEMO_DIR/server.py" ]]; then
  echo "Missing backend server: $DEMO_DIR/server.py" >&2
  exit 1
fi

if [[ ! -d "$DEMO_DIR/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  (cd "$DEMO_DIR" && npm install)
fi

find_backend_pids() {
  if command -v pgrep >/dev/null 2>&1; then
    pgrep -f "$DEMO_DIR/server.py" || true
    return
  fi
  ps -ef | grep "$DEMO_DIR/server.py" | grep -v grep | awk '{print $2}'
}

existing_pids="$(find_backend_pids)"
if [[ -n "$existing_pids" ]]; then
  echo "Stopping existing backend: $existing_pids"
  kill $existing_pids >/dev/null 2>&1 || true
  sleep 1
fi

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting backend on ${HOST_ADDR}:${BACKEND_PORT}..."
"$PYTHON_BIN" "$DEMO_DIR/server.py" --host "$HOST_ADDR" --port "$BACKEND_PORT" >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

sleep 1
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
  echo "Backend failed to start. Last log lines:" >&2
  tail -n 50 "$BACKEND_LOG" >&2 || true
  exit 1
fi

IP_ADDR="$(detect_ip)"
if [[ -z "$IP_ADDR" ]]; then
  IP_ADDR="127.0.0.1"
fi

echo "Frontend URL: http://${IP_ADDR}:${FRONTEND_PORT}"
echo "(If this IP is not reachable, set PUBLIC_IP or use SSH tunneling.)"

cd "$DEMO_DIR"
exec npm run dev -- --host "$HOST_ADDR" --port "$FRONTEND_PORT"
