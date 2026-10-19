#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DAPTA_ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export DAPTA_ROOT_DIR="$ROOT_DIR"
AGENT_ENV="${DAPTA_ASI1_ENV:-$ROOT_DIR/asi1/.env}"
PYTHON_BIN="${DAPTA_ASI1_PYTHON:-$ROOT_DIR/.venv-asi1/bin/python}"
ACCESS_FILE="${DAPTA_TRAINING_ACCESS_PATH:-$ROOT_DIR/artifacts/asi1-training-access.json}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

set -a
source "$AGENT_ENV"
set +a
STATE_PATH="${DAPTA_ASI1_STATE_PATH:-$ROOT_DIR/artifacts/asi1-agent.sqlite3}"
if [[ "$STATE_PATH" != /* ]]; then
  STATE_PATH="$ROOT_DIR/$STATE_PATH"
fi

if [[ "${1:-}" != "enable" && "${1:-}" != "disable" ]]; then
  echo "Usage: $0 enable [--ttl-seconds 600] | disable" >&2
  exit 2
fi

"$ROOT_DIR/asi1/stop.sh"
if output="$($PYTHON_BIN -m asi1.training_access --env "$AGENT_ENV" --state "$STATE_PATH" --access-file "$ACCESS_FILE" "$@")"; then
  "$ROOT_DIR/asi1/start.sh"
  printf '%s\n' "$output"
else
  "$ROOT_DIR/asi1/start.sh"
  exit 1
fi
