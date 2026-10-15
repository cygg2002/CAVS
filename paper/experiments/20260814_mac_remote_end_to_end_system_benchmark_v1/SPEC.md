# Mac-to-remote end-to-end system benchmark specification

## Goal

Measure the complete deployable edge-cloud path with the local Mac acting as
an Edge proxy and `111_wcy@10.193.114.121` acting as the Cloud. Quantify the
gap between the paper's offline payload/AP accounting and a real HTTP system.

## Frozen contract

- Dataset: deterministic subset of COCO val2017.
- Edge model: repository `yolo26n.pt`, SHA-256 pinned by the deployed policy.
- Cloud model: repository `yolo26x.pt`, SHA-256 pinned by the deployed policy.
- Learner: exported three-seed CAVS Current+MAUR package.
- Main deployable action space for this benchmark: Edge / Original / Q90.
- CAVS budgets: global Fixed-byte 10%, 25%, and 40%.
- Cloud input size: 640; detector confidence: 0.001.
- Transport: HTTP JSON metadata plus Base64 JPEG through the existing SSH tunnel.
- No GT, final reward, or cloud response enters the online action decision.

## Measurements

- input decode, Edge inference, Edge postprocess;
- CAVS candidate construction and exact JPEG cost computation;
- request JSON serialization and actual HTTP request-body bytes;
- client round trip;
- server JSON/Base64/JPEG decode, Cloud inference, postprocess, and serialization;
- response bytes, transport residual, fusion, and end-to-end latency;
- payload/capacity utilization, wire-byte inflation, failure rate, and actions;
- COCO AP50-95/AP50/AP75 on the measured subset.

## Boundaries

- This is a desktop-edge proxy benchmark, not an RK3588 latency claim.
- The shared server GPU load is recorded; measurements under contention are
  reported as observed system latency, not an idle-hardware lower bound.
- Existing paper experiments, demo processes, ports, and artifacts are read-only.
- The benchmark uses an isolated cloud port and output directory.
- ROI is excluded from the first main comparison because the current Oregon
  deployment selects no ROI and the stable cross-domain core is CAVS-OQ.

## Success criteria

- Model hashes and runtime settings match the frozen deployment contract.
- Warmups are excluded from measured rows.
- Each image has one unique record and every CAVS plan has one action per image.
- Raw JPEG request bytes stay within each offline Fixed-byte capacity.
- Actual HTTP body bytes, response bytes, and component timings are preserved.
- Cloud failure falls back to the Edge prediction and is counted.
- A machine-readable result and a Chinese theory-versus-real report are produced.

