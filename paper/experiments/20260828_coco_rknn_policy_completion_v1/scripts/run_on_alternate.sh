#!/usr/bin/env bash
set -euo pipefail

ROOT="${CAVS_POLICY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${CAVS_POLICY_PYTHON:-/home/linzhicheng/.conda/envs/yolo26/bin/python}"
VENDOR="${CAVS_POLICY_VENDOR:-/var/tmp/cavs_main32_20260829/vendor}"
STATE="$ROOT/runs/alternate_execution"

mkdir -p "$STATE"
exec 9>"$STATE/launcher.lock"
if ! flock -n 9; then
  echo "another alternate policy-completion launcher owns the lock" >&2
  exit 2
fi
if [[ ! -f "$ROOT/runs/alternate_relocation.json" ]]; then
  echo "alternate relocation gate is missing" >&2
  exit 3
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "configured Python is not executable: $PYTHON" >&2
  exit 4
fi

export PYTHONPATH="$VENDOR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$ROOT"

# Fail before launching either long-running branch if the isolated mirror is
# missing a dynamically imported training dependency.
"$PYTHON" - <<'PY'
import importlib.util
from pathlib import Path

from paper.experiments.CA_AORIC.ca_aoric import CocoMetricBackend
from paper.experiments.GAIR_v5.gair.fusion import class_aware_nms
from paper.experiments.RCMAR_v4.rcmar.coco_eval import internal_to_coco

assert CocoMetricBackend is not None
assert class_aware_nms is not None
assert internal_to_coco is not None

adapter_path = Path(
    "../20260810_oregon_cloudv3_full_scheduler_afpd_paper_v1/scripts/training_adapter.py"
).resolve()
spec = importlib.util.spec_from_file_location("alternate_training_adapter_gate", adapter_path)
if spec is None or spec.loader is None:
    raise ImportError(adapter_path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
sources = adapter._sources()
if len(sources) != 12:
    raise RuntimeError(f"training adapter source gate is incomplete: {sorted(sources)}")
PY

date --iso-8601=seconds > "$STATE/started_at.txt"

run_cavs() {
  nice -n 15 "$PYTHON" scripts/train_backend.py --backend fp16 --stage evaluate
  nice -n 15 "$PYTHON" scripts/export_policy_backend.py --backend fp16
}

run_external() {
  nice -n 15 "$PYTHON" scripts/train_external.py --stage all
}

run_cavs > "$STATE/cavs_oq.log" 2>&1 &
cavs_pid=$!
run_external > "$STATE/external.log" 2>&1 &
external_pid=$!
printf '%s\n' "$cavs_pid" > "$STATE/cavs_oq.pid"
printf '%s\n' "$external_pid" > "$STATE/external.pid"

status=0
wait "$cavs_pid" || status=$?
wait "$external_pid" || status=$?
if (( status != 0 )); then
  printf '%s\n' "$status" > "$STATE/failed_exit_code.txt"
  exit "$status"
fi

"$PYTHON" - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("runs")
required = [
    root / "backends/fp16/export/policy/manifest.json",
    root / "external/export/policy/manifest.json",
    root / "backends/fp16/evaluation/report.json",
    root / "external/evaluation/report.json",
]
verified = {}
for path in required:
    if not path.is_file():
        raise FileNotFoundError(path)
    json.loads(path.read_text(encoding="utf-8"))
    verified[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
Path("runs/alternate_execution/verified_outputs.json").write_text(
    json.dumps(verified, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
PY

date --iso-8601=seconds > "$STATE/completed_at.txt"
