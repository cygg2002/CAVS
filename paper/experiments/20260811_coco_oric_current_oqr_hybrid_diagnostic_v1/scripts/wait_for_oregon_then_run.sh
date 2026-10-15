#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO="$(cd "${ROOT}/../../.." && pwd)"
OREGON_MARKER="${REPO}/paper/experiments/20260810_oregon_cloudv3_full_scheduler_afpd_paper_v1/runs/report/completion.json"
EVALUATION_MARKER="${ROOT}/runs/evaluation/completion.json"
PYTHON="/home/111_wcy/anaconda3/envs/yolo26/bin/python"

echo "Waiting for Oregon and hybrid point evaluation."
while [[ ! -f "${OREGON_MARKER}" || ! -f "${EVALUATION_MARKER}" ]]; do
  sleep 30
done

echo "Prerequisites complete; starting paired bootstrap and report."
cd "${ROOT}"
exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 10 "${PYTHON}" scripts/run_experiment.py --stage all
