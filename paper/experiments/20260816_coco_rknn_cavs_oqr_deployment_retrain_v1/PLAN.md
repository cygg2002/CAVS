# Plan: RK3588 Deployment-Aligned CAVS-OQR

## Architecture Decisions

- The FP16 and INT8 paths are isolated by backend name and RKNN model SHA-256.
- RK3588 candidate collection is sequential per backend because the NPU is a
  shared measurement resource.  All CPU-only post-processing runs in parallel.
- Original/Q90 cloud predictions may be reused only after a model/protocol hash
  match; Current@640 predictions are rebuilt from each backend's own ROI.

## Tasks

- [x] Task 1: Define the frozen two-backend contract.
  - Acceptance: only `fp16` and `int8-heads` are accepted; a run has 18 jobs.
  - Verify: unit tests.

- [x] Task 2: Inspect the board bundle and define an exact candidate manifest.
  - Acceptance: each record contains model identity, image ID, image geometry,
    raw RKNN detections, and backend-derived ROI geometry; no policy scores.
  - Verify: host contract tests and board collector syntax check. A ten-image
    RKNN smoke run remains pending until the board input bundle is mounted.

- [x] Task 3: Import board candidates and rebuild backend-specific actions.
  - Acceptance: no missing/duplicate IDs; ROI geometry derives from matching
    backend detections; original-byte denominator remains native COCO JPEG.
  - Verify: 17 host/remote tests; six candidate SHA-256 checks; six common
    action bundles complete; reused Original/Q90 prediction hashes verified.

- [ ] Task 4: Generate rewards and train both backends. **In progress.**
  - Acceptance: 36 new jobs complete; source manifests prove no checkpoint
    reuse or cross-backend records.
  - Verify: learner reports and policy hashes.

- [ ] Task 5: Evaluate RKNN candidate caches and execute matching HTTP plans.
  - Acceptance: fixed-byte capacity is respected and every cloud request is
    either successful or explicitly recorded as a fallback.
  - Verify: per-backend result summaries and parity audit.

## Risks

| Risk | Mitigation |
|---|---|
| USB is unavailable to the Mac | Do not invent a board schema; wait for the bundle or board export. |
| NPU contention alters latency | Collect FP16 and INT8 sequentially. |
| Candidate-model mismatch | Require backend name and SHA-256 in every manifest. |
| Long cloud cache rebuild | Resume per role/backend and start FP16/INT8 host jobs concurrently after imports. |
