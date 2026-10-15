#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/../../../.." && pwd)"
PYTHON="${PYTHON:-/home/111_wcy/anaconda3/envs/yolo26/bin/python}"
LAUNCHER="$ROOT/runs/launcher"
mkdir -p "$LAUNCHER"

if [[ -f "$LAUNCHER/pid" ]]; then
  old_pid="$(tr -d '[:space:]' < "$LAUNCHER/pid")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "pipeline already running: $old_pid"
    exit 0
  fi
fi

cd "$REPO"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup nice -n 15 "$PYTHON" -u \
  "$ROOT/scripts/run_pipeline.py" --stage all \
  > "$LAUNCHER/pipeline.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$LAUNCHER/pid"
"$PYTHON" - "$LAUNCHER/launch.json" "$pid" <<'PY'
import json
from pathlib import Path
import socket
import sys
import time

path = Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "pid": int(sys.argv[2]),
            "host": socket.gethostname(),
            "launched_unix": time.time(),
            "stage": "all",
            "nice": 15,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
echo "started: $pid"

