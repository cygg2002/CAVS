#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/../../../.." && pwd)"
PYTHON="${PYTHON:-/Users/cygg_macmini/.venvs/python312/bin/python}"
REMOTE="111_wcy@10.193.114.121"
SSH_PORT="10022"
REMOTE_ROOT="/home/111_wcy/work/yolo26/ultralytics-main/paper/experiments/20260815_coco_cavs_oqr_perframe_deployment_retrain_v1"
WATCH="$ROOT/runs/watcher"
INPUTS="$ROOT/inputs"
mkdir -p "$WATCH" "$INPUTS"
echo "$$" > "$WATCH/pid"

while ! ssh -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE" \
  "test -f '$REMOTE_ROOT/runs/export/completion.json'"; do
  sleep 60
done

rsync -az -e "ssh -p $SSH_PORT -o BatchMode=yes" \
  "$REMOTE:$REMOTE_ROOT/runs/common/coco/final/samples.json" "$INPUTS/paper_samples.json"
rsync -az -e "ssh -p $SSH_PORT -o BatchMode=yes" \
  "$REMOTE:$REMOTE_ROOT/runs/evaluation/selections.json" "$INPUTS/remote_selections.json"
rsync -az -e "ssh -p $SSH_PORT -o BatchMode=yes" \
  "$REMOTE:$REMOTE_ROOT/runs/evaluation/results.csv" "$INPUTS/paper_results.csv"
rsync -az --delete -e "ssh -p $SSH_PORT -o BatchMode=yes" \
  "$REMOTE:$REMOTE_ROOT/runs/export/policy/" "$INPUTS/policy/"

"$PYTHON" "$ROOT/scripts/prepare_live_inputs.py"
curl -fsS --max-time 5 http://127.0.0.1:8767/health > "$WATCH/cloud_health.json"

cd "$REPO"
nice -n 15 "$PYTHON" -u "$ROOT/scripts/run_live.py" \
  --stage all --output "$ROOT/runs/live5000"

"$PYTHON" - "$WATCH/completion.json" <<'PY'
import json
from pathlib import Path
import sys
import time

Path(sys.argv[1]).write_text(
    json.dumps({"complete": True, "completed_unix": time.time()}, indent=2) + "\n",
    encoding="utf-8",
)
PY
