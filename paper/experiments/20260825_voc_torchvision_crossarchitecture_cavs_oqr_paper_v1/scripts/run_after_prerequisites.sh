#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="$ROOT/runs"
PYTHON="${VOC_PYTHON:-/home/111_wcy/anaconda3/envs/yolo26/bin/python}"

wait_for_completion() {
  local name="$1"
  local completion="$2"
  local pid_file="$3"

  while [[ ! -f "$completion" ]]; do
    if [[ -f "$pid_file" ]]; then
      local pid
      pid="$(<"$pid_file")"
      if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
        echo "[$name] producer exited without completion: pid=$pid" >&2
        return 1
      fi
    fi
    sleep 30
  done
  echo "[$name] completion observed"
}

cd "$ROOT"
wait_for_completion actions "$RUNS/actions/completion.json" "$ROOT/logs/actions.pid"
wait_for_completion rewards "$RUNS/rewards/completion.json" "$ROOT/logs/dev_cpu.pid"

"$PYTHON" scripts/evaluate_endpoints.py
"$PYTHON" scripts/build_common.py --role all
"$PYTHON" scripts/train_learners.py
"$PYTHON" scripts/evaluate_point.py
"$PYTHON" scripts/bootstrap.py
"$PYTHON" scripts/write_report.py

echo "[pipeline] all paper stages complete"
