#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
experiment_root=$(cd "$script_dir/.." && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
runs="$experiment_root/runs"
calibration_pid=${1:?"usage: $0 CALIBRATION_PID"}
poll_seconds=${POLL_SECONDS:-30}
analysis_python=/home/111_wcy/anaconda3/envs/yoloe/bin/python
runner="$experiment_root/scripts/run_experiment.py"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

cd "$repo_root"
log "waiting for calibration PID $calibration_pid"
while [[ ! -f "$runs/CALIBRATION_COMPLETED" ]]; do
    if ! kill -0 "$calibration_pid" 2>/dev/null; then
        log "calibration stopped without a completion marker"
        [[ -f "$runs/pipeline_status.json" ]] && tail -n 20 "$runs/pipeline_status.json"
        exit 1
    fi
    sleep "$poll_seconds"
done

if [[ -f "$runs/AUDIT_COMPLETED" ]]; then
    log "audit already completed"
    exit 0
fi

log "calibration frozen; starting audit"
if env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    nice -n 10 "$analysis_python" "$runner" --stage audit \
    > "$runs/audit.log" 2>&1; then
    log "audit completed"
else
    status=$?
    log "audit failed with exit status $status; see $runs/audit.log"
    exit "$status"
fi
