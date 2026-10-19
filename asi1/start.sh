#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DAPTA_ROOT_DIR:-/workspace/dapta}"
RUNTIME_ENV="${DAPTA_RUNTIME_ENV:-$ROOT_DIR/runtime.env}"
AGENT_ENV="${DAPTA_ASI1_ENV:-$ROOT_DIR/asi1/.env}"
PYTHON_BIN="${DAPTA_ASI1_PYTHON:-$ROOT_DIR/.venv-asi1/bin/python}"
LOG_FILE="$ROOT_DIR/artifacts/logs/asi1-agent.log"
PID_FILE="$ROOT_DIR/artifacts/pids/asi1-agent.pid"

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/artifacts/logs" "$ROOT_DIR/artifacts/pids"

set -a
source "$RUNTIME_ENV"
source "$AGENT_ENV"
set +a

AGENT_PORT="${DAPTA_AGENT_PORT:-8004}"
if [[ "$AGENT_PORT" == "${VLLM_PORT:-8003}" ]]; then
    echo "DAPTA_AGENT_PORT ($AGENT_PORT) must differ from VLLM_PORT; use 8004 for ASI:One." >&2
    exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(<"$PID_FILE")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "ASI:One companion is already running with PID $existing_pid"
    exit 0
  fi
fi

export PYTHONPATH="$ROOT_DIR"
nohup "$PYTHON_BIN" -m asi1.agent >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
echo "Started ASI:One companion with PID $(<"$PID_FILE")"
