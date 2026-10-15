#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
REPO=$(cd "$ROOT/../../.." && pwd)
PARENT="$REPO/paper/experiments/20260810_coco_afpd_forced_fullval_posthoc_v1"
MARKER="$PARENT/runs/bootstrap/completion.json"
PYTHON_BIN=${PYTHON_BIN:-/home/111_wcy/anaconda3/envs/yolo26/bin/python}

echo "[queued] waiting for parent: $MARKER"
while ! { [ -f "$MARKER" ] && rg -q '"complete": true' "$MARKER"; }; do
  sleep 30
done
echo "[ready] parent bootstrap complete"
exec "$PYTHON_BIN" "$ROOT/scripts/run_experiment.py" --stage all
