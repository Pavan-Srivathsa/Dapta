#!/usr/bin/env bash
# Operator-only smoke. Does not enable public ASI factory builds.
set -euo pipefail

ROOT_DIR="${DAPTA_ROOT_DIR:-/workspace/dapta}"
ENV_FILE="${DAPTA_RUNTIME_ENV:-$ROOT_DIR/runtime.env}"
PYTHON_BIN="${DAPTA_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
OUTPUT_DIR="$ROOT_DIR/artifacts/asi1-smoke/expense-line"

cd "$ROOT_DIR"
set -a
source "$ENV_FILE"
set +a

if [[ "${DAPTA_ASI1_ENABLE_FACTORY_BUILDS:-false}" == "true" ]]; then
  echo "Public factory builds are enabled. Disable them before this smoke run." >&2
  exit 2
fi

reload() {
  curl --fail --silent --show-error -X POST "http://127.0.0.1:8002/admin/load" \
    -H "Authorization: Bearer ${DAPTA_QWEN_TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{}' >/dev/null
}

curl --fail --silent --show-error -X POST "http://127.0.0.1:8002/admin/unload" \
  -H "Authorization: Bearer ${DAPTA_QWEN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{}' >/dev/null
trap reload EXIT

export PYTHONPATH="$ROOT_DIR/backend"
export DAPTA_ASI1_SMOKE_DIR="$OUTPUT_DIR"
"$PYTHON_BIN" "$ROOT_DIR/asi1/smoke_expense.py"
