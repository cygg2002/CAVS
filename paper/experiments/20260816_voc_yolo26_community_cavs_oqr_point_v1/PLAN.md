# Implementation plan

## Phase 1: protocol and assets

- [x] Pin downloaded checkpoint hashes and verify the 20 VOC class names.
- [ ] Extract canonical VOC archives and build stable COCO/YOLO role manifests.
- [ ] Verify exact role counts and identity disjointness.

## Phase 2: detector-action cache

- [ ] Cache Edge predictions and pooled Edge FPN features.
- [ ] Cache Cloud Original and Q90 predictions with exact payload bytes.
- [ ] Build Current@640 crops, map Cloud predictions, and create MAUR fusion.

## Phase 3: rewards and learners

- [ ] Build five fold-contained contextual reward contexts on fit only.
- [ ] Compute Original/Q90/ROI AP50-95 and AP50 rewards resumably.
- [ ] Train three CAVS seeds and all ORIC/MORIC seeds from scratch.
- [ ] Fit one SharedAction-OQR mapper on fit and freeze thresholds on calibration.

## Phase 4: point evaluation

- [ ] Evaluate Fixed-byte and Fixed-call 10%/25%/40% for all methods.
- [ ] Validate byte capacity, row uniqueness, and prediction identity alignment.
- [ ] Write CSV/JSON artifacts and a Chinese results report.

