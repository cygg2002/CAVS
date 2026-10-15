# COCO Causal Online HIL

This directory is an isolated, zero-lookahead experiment. It does not modify
the completed two-pass global HIL runs.

At each image the board performs RKNN Edge inference, builds the enabled JPEG
actions and exact byte counts, makes one causal decision, uploads the selected
payload over HTTP, fuses the response, and commits the frame before opening the
next image. The board never loads COCO annotations or a future-image plan.

## Method status

The current bundle contains:

- `cavs_oqr`: deployment-aligned three-seed CAVS-OQR, Current@640 + MAUR.
- `oric` and `moric`: frozen historical COCO source-style Edge/Original
  baselines, explicitly marked `score_adapted_historical_policy`.
- `confidence` and `random`: deterministic scalar controls.

`cavs_o` and `cavs_oq` are supported by the NumPy policy loader once their
independently retrained manifests are exported. The OQR manifest must not be
masked at runtime and relabelled as an independently trained O or OQ model.

The external manifest currently available in this bundle is a single frozen
ORIC/MORIC model pair. It is not a three-seed ensemble and it is not two
separately retrained AP50/AP50-95 policies. Those distinctions belong in the
paper table.

## Mac cloud service

Set a short-lived token in both shells. The cloud model hash must be the one
bound by the RKNN policy and run contract.

```bash
export CAVS_TOKEN='local-experiment-token'
python mac_server/cloud_server.py \
  --model /path/to/yolo26x.pt \
  --host 0.0.0.0 --port 8768 --device mps
```

Check the endpoint from the board network before the formal run:

```bash
curl -sf -H "Authorization: Bearer ${CAVS_TOKEN}" \
  http://MAC_IP:8768/health
```

## Board smoke and formal run

Run one method/budget at a time. Use a fresh `--run-id` for every method and
budget; the run contract and lock reject accidental reuse.

```bash
export CAVS_TOKEN='local-experiment-token'
export ONLINE_PYTHON=/usr/bin/python3

COMMON=(
  --sequence /path/to/coco_val2017_sequence.json
  --images-dir /path/to/coco_val2017/images
  --rknn /path/to/yolo26n-rk3588-fp16.rknn
  --rknn-helpers-dir /path/to/yolo26n_coco_val
  --cloud-url http://MAC_IP:8768/infer
  --cloud-sha256 9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92
  --token-env CAVS_TOKEN
  --core 0
)

# 50-image smoke
bash board_bundle/run_online.sh \
  --run-id smoke-cavs-oqr-10 --method cavs_oqr --budget 0.1 --limit 50 \
  --policy board_bundle/policies/cavs_oqr/manifest.json \
  --control board_bundle/controls/cavs_oqr/10.json "${COMMON[@]}"

# Formal one-method/one-budget run (remove --limit)
bash board_bundle/run_online.sh \
  --run-id formal-cavs-oqr-10 --method cavs_oqr --budget 0.1 \
  --policy board_bundle/policies/cavs_oqr/manifest.json \
  --control board_bundle/controls/cavs_oqr/10.json "${COMMON[@]}"
```

For `random`, pass the same seed used during calibration:

```bash
bash board_bundle/run_online.sh \
  --run-id formal-random-10-seed20260720 --method random \
  --random-seed 20260720 --budget 0.1 \
  --policy board_bundle/scripts/online_methods.py \
  --control board_bundle/controls/random/10.json "${COMMON[@]}"
```

Replace the method, policy and control path for `cavs_o`, `cavs_oq`, `oric`,
`moric`, or `confidence` after their method-specific controls have been
calibrated. Formal board runs must be sequential because latency is a system
measurement, not a CPU-throughput benchmark.

The runner also verifies `configs/rknn_helpers_manifest.json` against the
external `--rknn-helpers-dir` before loading RKNN preprocessing or decode code.
Do not replace those helper modules after a run has started.

The currently executable formal matrix has 15 runs: `cavs_oqr`, `oric`,
`moric`, `confidence`, and `random` at 10%, 25%, and 40%. Run a 50-image smoke
for each method family first, then execute those 15 runs sequentially. CAVS-O
and CAVS-OQ enter the matrix only after independent action-space training and
control calibration are complete.

## Host evaluation

Copy the completed `frames.jsonl` and its `run_contract.json` back to the host,
then validate the committed stream, compute COCO AP, and summarize latency:

```bash
python host/evaluate_online_run.py \
  --annotations /path/to/instances_val2017.json \
  --manifest board_bundle/sequences/coco_val2017_sequence.json \
  --frames runs/formal-cavs-oqr-10/frames.jsonl \
  --predictions-output runs/formal-cavs-oqr-10/predictions.json \
  --metrics-output runs/formal-cavs-oqr-10/metrics.json \
  --log-output runs/formal-cavs-oqr-10/coco_eval.log

python host/summarize_online_run.py \
  --frames runs/formal-cavs-oqr-10/frames.jsonl \
  --run-contract runs/formal-cavs-oqr-10/run_contract.json \
  --output runs/formal-cavs-oqr-10/system_summary.json
```

The report must show complete committed frames, zero prefix-budget violation,
zero unexplained cloud failures, action counts, actual JPEG/HTTP bytes, and
P50/P95/P99 timings. AP from this stream is the causal-online result; it must
not be mixed with the earlier global-selection result.
