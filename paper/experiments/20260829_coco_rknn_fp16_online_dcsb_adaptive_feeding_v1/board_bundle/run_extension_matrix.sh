#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ONLINE_PYTHON:?Set ONLINE_PYTHON to the board RKNN Python interpreter}"
: "${CAVS_TOKEN:?Set CAVS_TOKEN in the process environment}"
: "${CAVS_CLOUD_URL:?Set CAVS_CLOUD_URL to the binary-v2 /infer endpoint}"

TAG="${CAVS_FORMAL_TAG:-formal-dcsb-af-$(date +%Y%m%d-%H%M%S)}"
RUNS_ROOT="${CAVS_RUNS_ROOT:-/mnt/usb/rk3588/cavs_dcsb_af_20260829_runs}"
IMAGES_DIR="${CAVS_IMAGES_DIR:-/mnt/usb/rk3588/yolo26n_coco_val/coco_val2017/images}"
RKNN_PATH="${CAVS_RKNN_PATH:-/mnt/usb/rk3588/yolo26n_coco_val/models/yolo26n-rk3588-fp16.rknn}"
RKNN_HELPERS="${CAVS_RKNN_HELPERS:-/mnt/usb/rk3588/yolo26n_coco_val}"
CLOUD_SHA256="9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"
POLICY="$ROOT/policies/dcsb_adaptive_feeding/manifest.json"

run_case() {
  local index="$1"
  local method="$2"
  local label="$3"
  local budget="$4"
  local run_id
  run_id="${TAG}-r$(printf '%02d' "$index")-${method}-b${label}"
  bash "$ROOT/run_online.sh" \
    --run-id "$run_id" \
    --method "$method" \
    --budget "$budget" \
    --policy "$POLICY" \
    --control "$ROOT/controls/$method/$label.json" \
    --sequence "$ROOT/sequences/coco_val2017_sequence.json" \
    --images-dir "$IMAGES_DIR" \
    --rknn "$RKNN_PATH" \
    --rknn-helpers-dir "$RKNN_HELPERS" \
    --cloud-url "$CAVS_CLOUD_URL" \
    --cloud-sha256 "$CLOUD_SHA256" \
    --token-env CAVS_TOKEN \
    --core 0 \
    --runs-root "$RUNS_ROOT"
}

run_case 1 dcsb_budget 10 0.10
run_case 2 adaptive_feeding_budget 25 0.25
run_case 3 dcsb_budget 40 0.40
run_case 4 adaptive_feeding_native 100 1.0
run_case 5 adaptive_feeding_budget 10 0.10
run_case 6 dcsb_budget 25 0.25
run_case 7 adaptive_feeding_budget 40 0.40
run_case 8 dcsb_native 100 1.0

"$ONLINE_PYTHON" - "$RUNS_ROOT" "$TAG" <<'PY'
import json
import math
from pathlib import Path
import sys

root, tag = Path(sys.argv[1]), sys.argv[2]
runs = sorted(path for path in root.glob(f"{tag}-r*") if path.is_dir())
rows = []
for run in runs:
    frames = run / "frames.jsonl"
    records = [json.loads(line) for line in frames.open(encoding="utf-8")] if frames.is_file() else []
    cloud_failures = 0
    prefix_violations = 0
    invalid_boxes = 0
    payload_bytes = 0
    request_body_bytes = 0
    for record in records:
        decision = record["decision"]
        action = str(decision["action"])
        if action != "edge" and record.get("cloud", {}).get("ok") is not True:
            cloud_failures += 1
        payload_bytes += int(decision["payload_bytes"])
        request_body_bytes += int(record.get("cloud", {}).get("request_body_bytes", 0))
        state = record["controller_state"]
        if int(state["spent_bytes"]) * int(state["budget_denominator"]) > int(state["cumulative_original_bytes"]) * int(state["budget_numerator"]):
            prefix_violations += 1
        height, width = map(int, record["shape"])
        for box in record["resolved_boxes"]:
            x1, y1, x2, y2 = map(float, box["xyxy"])
            confidence = float(box["conf"])
            category = box["cls"]
            if not (
                all(map(math.isfinite, (x1, y1, x2, y2, confidence)))
                and 0.0 <= x1 < x2 <= width
                and 0.0 <= y1 < y2 <= height
                and 0.0 <= confidence <= 1.0
                and isinstance(category, int)
                and not isinstance(category, bool)
                and 0 <= category < 80
            ):
                invalid_boxes += 1
    wire_ratio = request_body_bytes / payload_bytes if payload_bytes else 0.0
    accepted = (
        len(records) == 5000
        and all(record.get("status") == "committed" for record in records)
        and cloud_failures == 0
        and prefix_violations == 0
        and invalid_boxes == 0
        and wire_ratio < 1.02
        and (run / "run_contract.json").is_file()
    )
    rows.append({
        "run_id": run.name,
        "frames": len(records),
        "contract": (run / "run_contract.json").is_file(),
        "cloud_failures": cloud_failures,
        "prefix_budget_violations": prefix_violations,
        "invalid_final_boxes": invalid_boxes,
        "jpeg_payload_bytes": payload_bytes,
        "http_request_body_bytes": request_body_bytes,
        "wire_payload_ratio": wire_ratio,
        "accepted": accepted,
    })
complete = len(rows) == 8 and all(row["accepted"] for row in rows)
summary = {"schema": "cavs_dcsb_af_board_completion_v1", "tag": tag, "complete": complete, "runs": rows}
destination = root / f"_{tag}_COMPLETE.json"
destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
raise SystemExit(0 if complete else 1)
PY
