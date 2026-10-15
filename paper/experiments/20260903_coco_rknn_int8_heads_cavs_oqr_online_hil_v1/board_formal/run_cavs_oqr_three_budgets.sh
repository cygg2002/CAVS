#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="$ROOT/board_bundle"
RUNS_ROOT="${RUNS_ROOT:-$ROOT/runs}"
PYTHON_BIN="${ONLINE_PYTHON:?set ONLINE_PYTHON to the RKNN runtime Python}"
CLOUD_URL="${CLOUD_URL:?set CLOUD_URL to the frozen cloud /infer endpoint}"
CLOUD_SHA256="9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"
TAG="${1:-formal-int8-heads-cavs-oqr-$(date +%Y%m%d%H%M%S)}"
RKNN="/mnt/usb/rk3588/yolo26n_coco_val/models/yolo26n-rk3588-int8-heads.rknn"
IMAGES="/mnt/usb/rk3588/yolo26n_coco_val/coco_val2017/images"

: "${CAVS_TOKEN:?CAVS_TOKEN must be set in the runtime environment}"

"$PYTHON_BIN" "$BUNDLE/scripts/verify_bundle.py" --root "$BUNDLE" --manifest "$BUNDLE/BUNDLE_MANIFEST.json"

run_case() {
  local pct="$1"
  local budget="$2"
  local run_id="${TAG}-${pct}"
  bash "$BUNDLE/run_online.sh" \
    --run-id "$run_id" --method cavs_oqr --budget "$budget" \
    --sequence "$BUNDLE/sequences/coco_val2017_sequence.json" \
    --images-dir "$IMAGES" \
    --policy "$BUNDLE/policies/cavs_oqr/manifest.json" \
    --control "$BUNDLE/controls/cavs_oqr/${pct}.json" \
    --rknn "$RKNN" --rknn-helpers-dir /mnt/usb/rk3588/yolo26n_coco_val \
    --cloud-url "$CLOUD_URL" --cloud-sha256 "$CLOUD_SHA256" \
    --token-env CAVS_TOKEN --runs-root "$RUNS_ROOT" --core 0 --progress-every 250
  "$PYTHON_BIN" - "$RUNS_ROOT/$run_id/frames.jsonl" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
if len(rows) != 5000 or [row["position"] for row in rows] != list(range(5000)):
    raise SystemExit("formal gate failed: incomplete or non-canonical frame sequence")
if any(row.get("status") != "committed" for row in rows):
    raise SystemExit("formal gate failed: uncommitted frame")
if any(row.get("execution_status") not in {"ok", "not_required"} for row in rows):
    raise SystemExit("formal gate failed: cloud fallback or invalid execution status")
num = int(rows[-1]["controller_state"]["budget_numerator"])
den = int(rows[-1]["controller_state"]["budget_denominator"])
violation = max(max(0, int(row["decision"]["spent_bytes"]) - int(row["decision"]["cumulative_original_bytes"]) * num // den) for row in rows)
if violation:
    raise SystemExit(f"formal gate failed: prefix byte violation={violation}")
payload = sum(int(row["decision"]["payload_bytes"]) for row in rows)
wire = sum(int(row.get("cloud", {}).get("request_body_bytes") or 0) for row in rows)
if payload and wire / payload > 1.02:
    raise SystemExit(f"formal gate failed: wire/payload={wire / payload:.6f}")
print(json.dumps({"frames": len(rows), "prefix_violation_bytes": violation, "payload_bytes": payload, "wire_bytes": wire}, sort_keys=True))
PY
  sync
}

run_case 10 0.10
run_case 25 0.25
run_case 40 0.40
