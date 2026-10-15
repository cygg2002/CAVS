# Spec: COCO CAVS-OQR paper-parity real-system rerun

## Objective

Rerun the 5,000-image COCO val2017 CAVS-Current-MAUR deployment after
removing the known protocol mismatches in the 20260814 benchmark. The frozen
learner, Edge model, Cloud model, dataset population, actions, budgets, and
paper point estimates must not change.

## Frozen method

- Learner: `CAVS-Current-MAUR`, three-seed exported ensemble.
- Actions: Edge, native Original JPEG, paper Q90, Current@640 ROI + MAUR.
- Edge: `yolo26n.pt`, SHA-256 pinned in `configs/protocol.json`.
- Cloud: `yolo26x.pt`, SHA-256 pinned in `configs/protocol.json`.
- Budgets: Fixed-byte 10%, 25%, and 40% of the native COCO JPEG total.
- Detector thresholds: `conf=0.001`, `iou=0.7`, `max_det=300`.

## Corrections relative to 20260814

1. Original transmits the source COCO JPEG bytes without Q95 re-encoding.
2. Q90 and ROI use Pillow JPEG `quality=90`, `optimize=True`,
   `subsampling=2`, matching the paper action cache.
3. Capacity is based on the native JPEG total of 814,705,164 bytes.
4. Edge, Original, and Q90 use `end2end=False`, validator-style
   `multi_label=True` NMS, and scale-down-only preprocessing. Current@640 ROI
   preserves the paper cache's FP16,
   `rect=True`, `end2end=False` predict semantics.
5. Every available action cost is checked against the frozen paper sample
   record. Original and Q90 mismatches are fatal. ROI bytes are fatal whenever
   the live ROI geometry equals the frozen geometry.
6. The Cloud health endpoint exposes the complete inference contract and the
   client rejects a mismatched server before uploading images.

## Interpretation boundary

The HTTP service processes one request at a time, whereas some paper caches
were generated in detector batches. The rerun aligns codec, head, thresholds,
precision, resize policy, and NMS. Any remaining batch-size or hardware
numeric difference is reported as deployment variance rather than learner
degradation.

## Success criteria

- Protocol regression tests pass.
- A smoke run completes without byte or health-contract violations.
- All required cloud requests succeed or are explicitly counted as fallbacks.
- Each budget consumes no more than its native-byte capacity.
- Real AP50-95 is within 0.005 of the frozen paper point at each budget, or the
  remaining gap is attributed with action-level parity evidence.
- The old deployment experiment and its results remain unchanged.
