#!/usr/bin/env bash
# Start the remote YOLO26x verifier with the named conda environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${DEMO_CONFIG:-$SCRIPT_DIR/demo.env}"
if [[ -f "$CONFIG_FILE" ]]; then
    # shellcheck disable=SC1090 -- the path is explicitly selected by the operator.
    source "$CONFIG_FILE"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-yolo26}"
if [[ -n "${YOLO26_PYTHON:-}" ]]; then
    PYTHON_BIN="$YOLO26_PYTHON"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV_NAME" && -x "${CONDA_PREFIX:-}/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    PYTHON_BIN="$CONDA_BASE/envs/$CONDA_ENV_NAME/bin/python"
else
    PYTHON_BIN=""
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Cannot find the $CONDA_ENV_NAME Python. Set YOLO26_PYTHON to its absolute path." >&2
    exit 1
fi

cd "$REPO_ROOT"
CLOUD_MODEL="${CLOUD_MODEL:-yolo26x.pt}"
if [[ ! -f "$CLOUD_MODEL" ]]; then
    echo "Cloud model not found: $REPO_ROOT/$CLOUD_MODEL" >&2
    exit 1
fi

echo "Starting cloud verifier with $PYTHON_BIN"
echo "Bind: ${CLOUD_HOST:-127.0.0.1}:${CLOUD_PORT:-8765}  Model: $CLOUD_MODEL  Device: ${CLOUD_DEVICE:-0}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/cloud_server.py" \
    --model "$CLOUD_MODEL" \
    --host "${CLOUD_HOST:-127.0.0.1}" \
    --port "${CLOUD_PORT:-8765}" \
    --imgsz "${CLOUD_IMGSZ:-640}" \
    --conf "${CLOUD_CONF:-0.001}" \
    --device "${CLOUD_DEVICE:-0}" \
    --min-imgsz "${CLOUD_MIN_IMGSZ:-320}" \
    --max-imgsz "${CLOUD_MAX_IMGSZ:-960}" \
    --max-request-mb "${CLOUD_MAX_REQUEST_MB:-20}"
