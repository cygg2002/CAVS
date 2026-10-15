#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/home/111_wcy/work/datasets/coco}"
DEVICE="${DEVICE:-0}"
RUN_TAG="${RUN_TAG:-coco_full_v2}"
EDGE_BATCH_SIZE="${EDGE_BATCH_SIZE:-8}"
CLOUD_BATCH_SIZE="${CLOUD_BATCH_SIZE:-8}"
ROI_BATCH_SIZE="${ROI_BATCH_SIZE:-16}"
HALF="${HALF:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_RULE_BASELINE="${SKIP_RULE_BASELINE:-0}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"

run_step() {
  local title="$1"
  shift
  echo
  echo "==== ${title} ===="
  echo "${PYTHON_EXE} $*"
  "${PYTHON_EXE}" "$@"
}

cd "${REPO_ROOT}"

TRAIN_LIST="${DATA_ROOT}/train2017.txt"
VAL_LIST="${DATA_ROOT}/val2017.txt"
POLICY_DIR="runs/edge_cloud/learned_policy_${RUN_TAG}"
POLICY_PATH="${REPO_ROOT}/${POLICY_DIR}/learned_upload_policy.json"

BASELINE_DIR="runs/edge_cloud/eval_${RUN_TAG}_baseline_metricfix"
RULE_DIR="runs/edge_cloud/eval_${RUN_TAG}_rule_aggressive"
LEARNED50_DIR="runs/edge_cloud/eval_${RUN_TAG}_learned_f050_r005_cloud512"
LEARNED60_DIR="runs/edge_cloud/eval_${RUN_TAG}_learned_f060_r008_cloud512"
LEARNED70_DIR="runs/edge_cloud/eval_${RUN_TAG}_learned_f070_r010_cloud512"
LEARNED80_DIR="runs/edge_cloud/eval_${RUN_TAG}_learned_f080_r010_cloud512"
SUMMARY_DIR="runs/edge_cloud/summary_${RUN_TAG}"
COST_DIR="runs/edge_cloud/cost_${RUN_TAG}"

echo "RepoRoot   : ${REPO_ROOT}"
echo "DataRoot   : ${DATA_ROOT}"
echo "TrainList  : ${TRAIN_LIST}"
echo "ValList    : ${VAL_LIST}"
echo "PolicyDir  : ${POLICY_DIR}"
echo "Device     : ${DEVICE}"
echo "EdgeBatch  : ${EDGE_BATCH_SIZE}"
echo "CloudBatch : ${CLOUD_BATCH_SIZE}"
echo "ROIBatch   : ${ROI_BATCH_SIZE}"
echo "Half       : ${HALF}"
echo
echo "Expected time on a single GPU:"
echo "- full train2017 learned uploader v2 labeling + fit: about 5 to 8 hours"
echo "- each val2017 evaluation run: about 20 to 35 minutes"
echo "- full pipeline with baseline + 4 learned runs + optional rule run: about 8 to 12 hours"

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  train_args=(
    examples/YOLO26-Edge-Cloud/train_learned_uploader.py
    --source "${TRAIN_LIST}"
    --edge-model yolo26s.pt
    --cloud-model yolo26x.pt
    --device "${DEVICE}"
    --edge-batch-size "${EDGE_BATCH_SIZE}"
    --cloud-batch-size "${CLOUD_BATCH_SIZE}"
    --roi-batch-size "${ROI_BATCH_SIZE}"
    --enable-adaptive-roi
    --roi-small-gain 1.6
    --roi-edge-gain 1.45
    --roi-large-gain 1.15
    --roi-extra-cloud-conf 0.55
    --roi-veto-edge-conf 0.25
    --max-roi-candidates 2
    --feature-noise-std 0.02
    --gain-normalizer labels
    --save-dir "${POLICY_DIR}"
  )
  if [[ "${HALF}" == "1" ]]; then
    train_args+=(--half)
  fi
  run_step "Train learned uploader v2 on COCO train2017" "${train_args[@]}"
fi

if [[ ! -f "${POLICY_PATH}" ]]; then
  echo "Learned policy not found: ${POLICY_PATH}" >&2
  exit 1
fi

run_step "Evaluate corrected baselines + oracle full-frame upper bound on COCO val2017" \
  examples/YOLO26-Edge-Cloud/evaluate_edge_cloud.py \
  --source "${VAL_LIST}" \
  --edge-model yolo26s.pt \
  --cloud-model yolo26x.pt \
  --device "${DEVICE}" \
  --conf 0.25 \
  --cloud-conf 0.25 \
  --metric-conf 0.001 \
  --metric-cloud-conf 0.001 \
  --eval-cloud-only \
  --eval-full-frame-fusion-upper \
  --save-dir "${BASELINE_DIR}"

if [[ "${SKIP_RULE_BASELINE}" != "1" ]]; then
  run_step "Evaluate rule aggressive baseline on COCO val2017" \
    examples/YOLO26-Edge-Cloud/evaluate_edge_cloud.py \
    --source "${VAL_LIST}" \
    --edge-model yolo26s.pt \
    --cloud-model yolo26x.pt \
    --device "${DEVICE}" \
    --conf 0.25 \
    --cloud-conf 0.25 \
    --metric-conf 0.001 \
    --metric-cloud-conf 0.001 \
    --upload-threshold 0.35 \
    --max-roi-per-image 1 \
    --keyframe-interval 30 \
    --enable-adaptive-roi \
    --roi-small-gain 1.6 \
    --roi-edge-gain 1.45 \
    --roi-large-gain 1.15 \
    --roi-extra-cloud-conf 0.55 \
    --roi-veto-edge-conf 0.25 \
    --enable-complexity-probe \
    --complexity-threshold 0.10 \
    --empty-complexity-threshold 0.05 \
    --complexity-max-detections 3 \
    --complexity-coverage-threshold 0.06 \
    --save-dir "${RULE_DIR}"
fi

run_step "Evaluate learned uploader f0.50 r0.05 cloud512 on COCO val2017" \
  examples/YOLO26-Edge-Cloud/evaluate_edge_cloud.py \
  --source "${VAL_LIST}" \
  --edge-model yolo26s.pt \
  --cloud-model yolo26x.pt \
  --device "${DEVICE}" \
  --conf 0.25 \
  --cloud-conf 0.25 \
  --metric-conf 0.001 \
  --metric-cloud-conf 0.001 \
  --cloud-imgsz 512 \
  --enable-learned-policy \
  --learned-policy-model "${POLICY_PATH}" \
  --learned-frame-threshold 0.50 \
  --learned-roi-threshold 0.05 \
  --learned-cost-weight 0.02 \
  --learned-max-roi-per-image 1 \
  --enable-adaptive-roi \
  --roi-small-gain 1.6 \
  --roi-edge-gain 1.45 \
  --roi-large-gain 1.15 \
  --roi-extra-cloud-conf 0.55 \
  --roi-veto-edge-conf 0.25 \
  --save-dir "${LEARNED50_DIR}"

run_step "Evaluate learned uploader f0.60 r0.08 cloud512 on COCO val2017" \
  examples/YOLO26-Edge-Cloud/evaluate_edge_cloud.py \
  --source "${VAL_LIST}" \
  --edge-model yolo26s.pt \
  --cloud-model yolo26x.pt \
  --device "${DEVICE}" \
  --conf 0.25 \
  --cloud-conf 0.25 \
  --metric-conf 0.001 \
  --metric-cloud-conf 0.001 \
  --cloud-imgsz 512 \
  --enable-learned-policy \
  --learned-policy-model "${POLICY_PATH}" \
  --learned-frame-threshold 0.60 \
  --learned-roi-threshold 0.08 \
  --learned-cost-weight 0.02 \
  --learned-max-roi-per-image 1 \
  --enable-adaptive-roi \
  --roi-small-gain 1.6 \
  --roi-edge-gain 1.45 \
  --roi-large-gain 1.15 \
  --roi-extra-cloud-conf 0.55 \
  --roi-veto-edge-conf 0.25 \
  --save-dir "${LEARNED60_DIR}"

run_step "Evaluate learned uploader f0.70 r0.10 cloud512 on COCO val2017" \
  examples/YOLO26-Edge-Cloud/evaluate_edge_cloud.py \
  --source "${VAL_LIST}" \
  --edge-model yolo26s.pt \
  --cloud-model yolo26x.pt \
  --device "${DEVICE}" \
  --conf 0.25 \
  --cloud-conf 0.25 \
  --metric-conf 0.001 \
  --metric-cloud-conf 0.001 \
  --cloud-imgsz 512 \
  --enable-learned-policy \
  --learned-policy-model "${POLICY_PATH}" \
  --learned-frame-threshold 0.70 \
  --learned-roi-threshold 0.10 \
  --learned-cost-weight 0.02 \
  --learned-max-roi-per-image 1 \
  --enable-adaptive-roi \
  --roi-small-gain 1.6 \
  --roi-edge-gain 1.45 \
  --roi-large-gain 1.15 \
  --roi-extra-cloud-conf 0.55 \
  --roi-veto-edge-conf 0.25 \
  --save-dir "${LEARNED70_DIR}"

run_step "Evaluate learned uploader f0.80 r0.10 cloud512 on COCO val2017" \
  examples/YOLO26-Edge-Cloud/evaluate_edge_cloud.py \
  --source "${VAL_LIST}" \
  --edge-model yolo26s.pt \
  --cloud-model yolo26x.pt \
  --device "${DEVICE}" \
  --conf 0.25 \
  --cloud-conf 0.25 \
  --metric-conf 0.001 \
  --metric-cloud-conf 0.001 \
  --cloud-imgsz 512 \
  --enable-learned-policy \
  --learned-policy-model "${POLICY_PATH}" \
  --learned-frame-threshold 0.80 \
  --learned-roi-threshold 0.10 \
  --learned-cost-weight 0.02 \
  --learned-max-roi-per-image 1 \
  --enable-adaptive-roi \
  --roi-small-gain 1.6 \
  --roi-edge-gain 1.45 \
  --roi-large-gain 1.15 \
  --roi-extra-cloud-conf 0.55 \
  --roi-veto-edge-conf 0.25 \
  --save-dir "${LEARNED80_DIR}"

if [[ "${SKIP_SUMMARY}" != "1" ]]; then
  summary_runs=("${BASELINE_DIR}")
  if [[ "${SKIP_RULE_BASELINE}" != "1" ]]; then
    summary_runs+=("${RULE_DIR}")
  fi
  summary_runs+=("${LEARNED50_DIR}" "${LEARNED60_DIR}" "${LEARNED70_DIR}" "${LEARNED80_DIR}")

  run_step "Summarize COCO full runs" \
    examples/YOLO26-Edge-Cloud/summarize_strategy_results.py \
    "${summary_runs[@]}" \
    --save-dir "${SUMMARY_DIR}"

  run_step "Analyze COCO full run costs" \
    examples/YOLO26-Edge-Cloud/analyze_edge_cloud_costs.py \
    --summary-csv "${SUMMARY_DIR}/strategy_summary.csv" \
    --save-dir "${COST_DIR}" \
    --edge-model yolo26s.pt \
    --cloud-model yolo26x.pt \
    --learned-policy "${POLICY_PATH}" \
    --imgsz 640
fi

echo
echo "Pipeline completed."
echo "Policy : ${POLICY_PATH}"
echo "Summary: ${SUMMARY_DIR}/strategy_summary.csv"
echo "Cost   : ${COST_DIR}/cost_analysis.csv"
