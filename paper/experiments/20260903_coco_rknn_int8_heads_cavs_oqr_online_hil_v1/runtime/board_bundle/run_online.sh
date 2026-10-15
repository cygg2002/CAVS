#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ONLINE_PYTHON:-python3}"

"${PYTHON_BIN}" "${BUNDLE_DIR}/scripts/verify_bundle.py" \
  --root "${BUNDLE_DIR}" \
  --manifest "${BUNDLE_DIR}/BUNDLE_MANIFEST.json"

exec "${PYTHON_BIN}" "${BUNDLE_DIR}/scripts/online_runner.py" \
  --bundle-manifest "${BUNDLE_DIR}/BUNDLE_MANIFEST.json" "$@"
