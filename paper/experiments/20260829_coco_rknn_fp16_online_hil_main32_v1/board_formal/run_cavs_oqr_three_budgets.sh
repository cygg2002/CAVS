#!/usr/bin/env bash
set -euo pipefail

ROOT="${CAVS_ROOT:-/home/orangepi/cavs_online_hil_final_v2_smoke_20260828}"
BUNDLE="$ROOT/board_bundle"
RUNS_ROOT="${RUNS_ROOT:-$ROOT/runs}"
ONLINE_PYTHON="${ONLINE_PYTHON:-$HOME/venvs/rknn-yolo26/bin/python}"
CLOUD_URL="${CLOUD_URL:-http://10.191.0.59:8768/infer}"
CLOUD_SHA256="${CLOUD_SHA256:-9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92}"
TAG="${1:-formal-v2-altserver-cavs-oqr-$(date +%Y%m%d-%H%M%S)}"

: "${CAVS_TOKEN:?CAVS_TOKEN must be set in the runtime environment}"

"$ONLINE_PYTHON" "$BUNDLE/scripts/verify_bundle.py" \
  --root "$BUNDLE" \
  --manifest "$BUNDLE/BUNDLE_MANIFEST.json"

run_case() {
  local pct="$1"
  local budget="$2"
  local run_id="${TAG}-${pct}"
  local run_dir="$RUNS_ROOT/$run_id"

  printf 'START run_id=%s budget=%s time=%s\n' "$run_id" "$budget" "$(date -Is)"
  bash "$BUNDLE/run_online.sh" \
    --run-id "$run_id" \
    --method cavs_oqr \
    --budget "$budget" \
    --sequence "$BUNDLE/sequences/coco_val2017_sequence.json" \
    --images-dir /mnt/usb/rk3588/yolo26n_coco_val/coco_val2017/images \
    --policy "$BUNDLE/policies/cavs_oqr/manifest.json" \
    --control "$BUNDLE/controls/cavs_oqr/${pct}.json" \
    --rknn /mnt/usb/rk3588/yolo26n_coco_val/models/yolo26n-rk3588-fp16.rknn \
    --rknn-helpers-dir /mnt/usb/rk3588/yolo26n_coco_val \
    --cloud-url "$CLOUD_URL" \
    --cloud-sha256 "$CLOUD_SHA256" \
    --token-env CAVS_TOKEN \
    --runs-root "$RUNS_ROOT" \
    --core 0 \
    --progress-every 250

  "$ONLINE_PYTHON" - "$run_dir/frames.jsonl" <<'PY'
import json
import sys

path = sys.argv[1]
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
if len(rows) != 5000:
    raise SystemExit(f"formal gate failed: expected 5000 frames, got {len(rows)}")
if [row["position"] for row in rows] != list(range(5000)):
    raise SystemExit("formal gate failed: frame positions are not the canonical prefix")
if any(row.get("status") != "committed" for row in rows):
    raise SystemExit("formal gate failed: uncommitted frame")
if any(row.get("execution_status") not in {"ok", "not_required"} for row in rows):
    raise SystemExit("formal gate failed: cloud fallback or invalid execution status")

numerator = int(rows[-1]["controller_state"]["budget_numerator"])
denominator = int(rows[-1]["controller_state"]["budget_denominator"])
maximum_violation = max(
    max(
        0,
        int(row["decision"]["spent_bytes"])
        - (int(row["decision"]["cumulative_original_bytes"]) * numerator // denominator),
    )
    for row in rows
)
if maximum_violation:
    raise SystemExit(f"formal gate failed: prefix violation={maximum_violation}")

payload = sum(int(row["decision"]["payload_bytes"]) for row in rows)
body = sum(int(row.get("cloud", {}).get("request_body_bytes") or 0) for row in rows)
ratio = body / payload if payload else 1.0
if ratio > 1.02:
    raise SystemExit(f"formal gate failed: HTTP-body/payload ratio={ratio:.6f}")

print(
    json.dumps(
        {
            "frames": len(rows),
            "cloud_failures": 0,
            "maximum_prefix_violation_bytes": maximum_violation,
            "selected_payload_bytes": payload,
            "wire_request_body_bytes": body,
            "wire_to_payload_ratio": ratio,
        },
        sort_keys=True,
    )
)
PY
  sync
  printf 'PASS run_id=%s time=%s\n' "$run_id" "$(date -Is)"
}

run_case 10 0.10
run_case 25 0.25
run_case 40 0.40
printf 'ALL_COMPLETE tag=%s time=%s\n' "$TAG" "$(date -Is)"
