#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${RKNN_PYTHON:-${HOME}/venvs/rknn-yolo26/bin/python}"
MODEL="fp16"
STAGE="all"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
CORE="0"
LIMIT="0"
CLOUD_URL=""
CLOUD_TIMEOUT="60"
TOKEN_ENV="CAVS_TOKEN"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:?--model requires fp16}"; shift 2 ;;
    --stage) STAGE="${2:?--stage requires a value}"; shift 2 ;;
    --run-id) RUN_ID="${2:?--run-id requires a value}"; shift 2 ;;
    --core) CORE="${2:?--core requires a value}"; shift 2 ;;
    --limit) LIMIT="${2:?--limit requires a value}"; shift 2 ;;
    --cloud-url) CLOUD_URL="${2:?--cloud-url requires a value}"; shift 2 ;;
    --cloud-timeout) CLOUD_TIMEOUT="${2:?--cloud-timeout requires a value}"; shift 2 ;;
    --token-env) TOKEN_ENV="${2:?--token-env requires a value}"; shift 2 ;;
    -h|--help)
      echo "Usage: bash run_board.sh [--model fp16] [--stage candidates|plan|upload|evaluate|all] [--limit N] --cloud-url http://HOST:8767/infer"
      exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: RKNN Python not found: ${PYTHON_BIN}; set RKNN_PYTHON." >&2
  exit 1
fi
case "${MODEL}" in fp16) ;; *) echo "ERROR: this bundle is bound to fp16" >&2; exit 2 ;; esac
case "${STAGE}" in candidates|plan|upload|evaluate|all) ;; *) echo "ERROR: invalid --stage" >&2; exit 2 ;; esac

if [[ "${STAGE}" == "evaluate" || "${STAGE}" == "all" ]]; then
  if ! "${PYTHON_BIN}" -c 'import pycocotools' >/dev/null 2>&1; then
    WHEELS_DIR="${BUNDLE_DIR}/../yolo26n_coco_val/wheels"
    "${PYTHON_BIN}" -m pip install --no-index --find-links "${WHEELS_DIR}" pycocotools==2.0.11
  fi
fi

"${PYTHON_BIN}" "${BUNDLE_DIR}/scripts/verify_bundle.py"
"${PYTHON_BIN}" "${BUNDLE_DIR}/scripts/board_client.py" \
  --model fp16 --stage "${STAGE}" --run-id "${RUN_ID}" --core "${CORE}" --limit "${LIMIT}" \
  --cloud-url "${CLOUD_URL}" --cloud-timeout "${CLOUD_TIMEOUT}" --token-env "${TOKEN_ENV}"
echo "CAVS_RUN_ID=${RUN_ID}"
