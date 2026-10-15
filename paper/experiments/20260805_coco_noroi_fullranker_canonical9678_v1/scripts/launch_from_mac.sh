#!/usr/bin/env bash
set -euo pipefail

LOCAL_REPO="/Users/cygg_macmini/work/Codex/ultralytics-main"
REMOTE_HOST="111_wcy@10.193.114.121"
REMOTE_PORT="10022"
REMOTE_REPO="/home/111_wcy/work/yolo26/ultralytics-main"
EXPERIMENT="paper/experiments/20260805_coco_noroi_fullranker_canonical9678_v1"
REMOTE_PYTHON="/home/111_wcy/anaconda3/envs/yolo26/bin/python"

rsync -az \
  --exclude '__pycache__' \
  --exclude 'runs' \
  -e "ssh -p ${REMOTE_PORT} -o BatchMode=yes" \
  "${LOCAL_REPO}/${EXPERIMENT}/" \
  "${REMOTE_HOST}:${REMOTE_REPO}/${EXPERIMENT}/"

ssh -p "${REMOTE_PORT}" -o BatchMode=yes "${REMOTE_HOST}" \
  "cd '${REMOTE_REPO}' && \
   '${REMOTE_PYTHON}' -m unittest discover -s '${EXPERIMENT}/tests' -v && \
   '${REMOTE_PYTHON}' '${EXPERIMENT}/scripts/run_noroi.py' --stage preflight"

ssh -p "${REMOTE_PORT}" -o BatchMode=yes "${REMOTE_HOST}" \
  "cd '${REMOTE_REPO}' || exit 1; \
   mkdir -p '${EXPERIMENT}/runs'; \
   nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
     '${REMOTE_PYTHON}' '${EXPERIMENT}/scripts/run_noroi.py' --stage all \
     > '${EXPERIMENT}/runs/pipeline.log' 2>&1 < /dev/null & \
   pid=\$!; \
   echo \"\${pid}\" > '${EXPERIMENT}/runs/pipeline.pid'; \
   printf '{\"launched\":true,\"pid\":%s,\"workers_point\":32,\"workers_bootstrap\":32}\n' \"\${pid}\" \
     > '${EXPERIMENT}/runs/launch_status.json'; \
   echo \"PID=\${pid}\""

echo "Launched ${EXPERIMENT} on ${REMOTE_HOST}."
