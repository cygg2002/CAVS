#!/usr/bin/env bash
# Preflight and start the local edge UI with the named conda environment.

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
EDGE_MODEL="${EDGE_MODEL:-yolo26n.pt}"
POLICY_PATH="${POLICY_PATH:-$SCRIPT_DIR/policies/current_maur_v2/manifest.json}"
CLOUD_URL="${CLOUD_URL:-http://127.0.0.1:8765/infer}"

PREFLIGHT_ARGS=(
    preflight
    --edge-model "$EDGE_MODEL"
    --policy-path "$POLICY_PATH"
    --cloud-url "$CLOUD_URL"
    --cloud-timeout "${CLOUD_TIMEOUT:-8.0}"
)
if [[ "${REQUIRE_CLOUD:-0}" == "1" ]]; then
    PREFLIGHT_ARGS+=(--require-cloud)
fi
"$PYTHON_BIN" "$SCRIPT_DIR/demo_system.py" "${PREFLIGHT_ARGS[@]}"

UI_ARGS=(
    --edge-model "$EDGE_MODEL"
    --cloud-url "$CLOUD_URL"
    --policy-path "$POLICY_PATH"
    --device "${EDGE_DEVICE:-}"
    --server-name "${UI_HOST:-127.0.0.1}"
    --server-port "${UI_PORT:-7860}"
    --save-dir "${SAVE_DIR:-runs/edge_cloud/gradio_demo}"
    --max-file-size "${UI_MAX_FILE_SIZE:-512mb}"
    --cloud-timeout "${CLOUD_TIMEOUT:-8.0}"
    --cloud-imgsz "${CLOUD_IMGSZ:-640}"
    --online-fixed-byte-budget "${ONLINE_FIXED_BYTE_BUDGET:-0.1}"
)
if [[ "${DISABLE_ONLINE_BUDGET:-0}" == "1" ]]; then
    UI_ARGS+=(--disable-online-budget)
fi
if [[ "${OPEN_BROWSER:-1}" == "1" ]]; then
    UI_ARGS+=(--inbrowser)
fi

echo "Starting local edge UI at http://${UI_HOST:-127.0.0.1}:${UI_PORT:-7860}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/gradio_demo.py" "${UI_ARGS[@]}"
