#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${CLOUD_PYTHON:-python3}"
exec "${PYTHON_BIN}" "${BUNDLE_DIR}/cloud_server.py" "$@"
