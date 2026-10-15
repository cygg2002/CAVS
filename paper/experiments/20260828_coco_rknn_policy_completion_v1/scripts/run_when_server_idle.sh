#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/111_wcy/work/yolo26/ultralytics-main
EXP="$ROOT/paper/experiments/20260828_coco_rknn_policy_completion_v1"
PY=/home/111_wcy/anaconda3/envs/yolo26/bin/python
STATE="$EXP/runs/queue"
mkdir -p "$STATE"

exec 9>"$STATE/launcher.lock"
if ! flock -n 9; then
  echo "another policy-completion launcher already owns the lock"
  exit 2
fi

echo "queue_started=$(date --iso-8601=seconds)"
while :; do
  active_bootstrap="$(pgrep -fc '20260828_cavs_final_paper_server_completion_v1/scripts/accelerate_bootstrap.py' || true)"
  load_one="$(awk '{print int($1)}' /proc/loadavg)"
  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  if (( active_bootstrap == 0 && load_one < 20 && available_kib > 33554432 )); then
    break
  fi
  echo "queue_wait=$(date --iso-8601=seconds) bootstrap=$active_bootstrap load1=$load_one mem_available_kib=$available_kib"
  sleep 60
done

echo "execution_started=$(date --iso-8601=seconds)"
cd "$EXP"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

nice -n 15 "$PY" scripts/train_backend.py --backend fp16 --stage all
nice -n 15 "$PY" scripts/export_policy_backend.py --backend fp16
nice -n 15 "$PY" scripts/train_external.py --stage all

"$PY" - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("runs")
outputs = [
    root / "backends/fp16/export/policy/manifest.json",
    root / "external/export/policy/manifest.json",
    root / "backends/fp16/evaluation/report.json",
    root / "external/evaluation/report.json",
]
for path in outputs:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"verified_output={path} sha256={digest}")
PY

date --iso-8601=seconds > "$STATE/completed_at.txt"
echo "execution_completed=$(cat "$STATE/completed_at.txt")"
