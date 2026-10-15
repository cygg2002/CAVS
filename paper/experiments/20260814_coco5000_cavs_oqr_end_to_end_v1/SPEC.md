# COCO5000 CAVS-OQR end-to-end system benchmark specification

## Objective

Replay all 5,000 COCO val2017 images through the real Mac-to-remote edge-cloud
demo path using the frozen three-seed CAVS-OQR learner. Measure full-set AP,
actual request bytes, failures, and component latency for Fixed-byte 10%, 25%,
and 40%.

## Frozen runtime contract

- Edge proxy: local Mac, `yolo26n.pt`, MPS, imgsz=640, conf=0.001.
- Cloud: remote RTX 4090, hash-pinned `yolo26x.pt`, imgsz=640, conf=0.001.
- Learner: exported `CAVS-Current-MAUR (3-seed)` package.
- Actions: Edge, Original Q95, Q90, Current@640 ROI Q90.
- ROI fusion: frozen MAUR duplicate removal and class-aware NMS.
- Budgets: global raw-JPEG Fixed-byte 10%, 25%, and 40%.
- Transport: loopback HTTP through an SSH tunnel, JSON + Base64 JPEG.
- Dataset: all canonical COCO val2017 JPEG files, sorted by image id.

## Execution model

1. Run all 5,000 Edge detections and construct exact O/Q/R candidates.
2. Freeze one global CAVS-OQR plan for each budget.
3. Upload only the union of non-Edge image-action pairs selected by the plans.
4. Replace full-frame actions with Cloud output; apply MAUR for ROI actions.
5. Evaluate Edge and the three CAVS-OQR outputs with standard COCO bbox AP.

Candidates and requests are append-only JSONL checkpoints. A restart skips every
completed candidate and successful request, so it does not repeat cloud work.

## Boundaries

- This is a post-hoc full-val deployment benchmark, not a new hidden final.
- Original uses the current demo's Q95 re-encoding, which may differ from cached
  paper Original semantics and is part of the deployment-gap measurement.
- The server GPU is shared. Latency represents observed contention, not idle 4090.
- Existing demo port 8765 and paper experiments remain unchanged.
- No GT, reward, or Cloud output enters candidate scoring or scheduling.

## Success criteria

- Exactly 5,000 unique candidate records and image ids.
- Every budget selects exactly one action per image and respects raw-JPEG capacity.
- Every required selected upload has a successful response or a counted fallback.
- ROI requests preserve crop geometry and Cloud boxes are mapped to full-image coordinates.
- Full-set AP50-95/AP50/AP75, action counts, payload/body bytes, and latency summaries exist.
- The report states the delta from the cached theoretical Current+MAUR point without
  presenting deployment and cached byte semantics as identical.

