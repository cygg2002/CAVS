# VOC community-YOLO26 CAVS-OQR point benchmark

## Objective

Run one isolated, reproducible PASCAL VOC point benchmark using the user-provided
community checkpoints:

- Edge: `yolo26n-voc.pt`;
- Cloud: `yolo26x-voc.pt`.

Every upload learner is retrained from scratch on VOC. The primary comparison is
`CAVS-OQR-MAUR` against ORIC/MORIC, Confidence, DCSB, and Random under the same
Fixed-byte and Fixed-call budgets. Bootstrap is intentionally out of scope.

## Dataset and roles

- Canonical detector-training population: VOC2007 trainval + VOC2012 trainval,
  16,551 images.
- Router fit: 15,629 images.
- Router calibration: 922 images.
- Point evaluation: VOC2007 test, 4,952 images.
- Fit/calibration are a deterministic hash partition of the 16,551-image
  population. VOC2007 test is never used to fit or calibrate a router.

The downloaded detector checkpoints were selected by their publisher using the
Ultralytics VOC validation split, which is VOC2007 test. Therefore this is a
post-hoc community-detector benchmark, not a new hidden confirmatory test.

## Frozen actions

- Edge: local YOLO26n result, zero upload bytes.
- Original: full JPEG sent to YOLO26x at 640.
- Q90: full image re-encoded as JPEG quality 90, sent to YOLO26x at 640.
- `current@640`: merged ROI generated from Edge detections using the frozen
  Current profile, JPEG quality 90, YOLO26x at 640, fused with Edge by MAUR.

All predictions use `conf=0.001`, `iou=0.7`, `max_det=300`, `rect=true`, and
`end2end=false`. Current ROI geometry uses `edge_conf=0.25` and maximum area
ratio 0.65.

## Learners

- `CAVS-OQR-MAUR`: CurrentActionValue, three independently trained seeds and a
  score ensemble.
- `CAVS-OQ-from-OQR`: hard OQ mask of the same CAVS ensemble, reported only as
  an internal ROI attribution control.
- `ORIC-AP5095`, `MORIC-AP5095`: primary external baselines, three seeds.
- `ORIC-AP50`, `MORIC-AP50`: reward-metric sensitivity baselines, three seeds.
- `DCSB-Score-Adapted`, `Confidence`, and Random with 50 frozen seeds.

Scalar rankers share one fit-trained, calibration-frozen SharedAction-OQR mapper.
No learner checkpoint from COCO, Oregon, or another detector is reused.

## Training target

CAVS and ORIC/MORIC use fold-contained contextual rewards with context size 32.
The primary target is contextual AP50-95 gain; AP50 is retained only for the
ORIC/MORIC sensitivity rows. Five deterministic folds prevent a target image
from appearing in its own reward context.

## Evaluation

- Primary: Fixed-byte budgets 10%, 25%, and 40% of all-Original upload bytes.
- Secondary: Fixed-call budgets 10%, 25%, and 40% of images.
- Metrics: AP50-95, AP50, AP75, KB/image, selected image count, action counts,
  requested bytes, capacity, and utilization.
- Point estimates only. No bootstrap, confidence interval, or significance claim.

## Boundaries

- Always: pin checkpoint hashes, preserve role identity, retrain every learner,
  use one evaluator and one capacity solver, write resumable atomic artifacts.
- Never: use VOC2007 test for reward, learner fitting, SharedAction fitting, or
  threshold selection; reuse COCO/Oregon learner scores; exceed byte capacity.
- Source experiments are read-only. All new output is under this experiment.

## Success criteria

- Checkpoint hashes and 20-class names pass preflight.
- Role counts are exactly 15,629 / 922 / 4,952 and identities are disjoint.
- Edge, Original, Q90, and Current@640+MAUR caches cover every role image.
- Contextual reward artifacts cover every fit image and cloud action.
- Three CAVS seeds and three seeds for each ORIC/MORIC variant complete.
- Every requested method/budget/mode row is present and byte-safe.
- `bootstrap_performed=false` is recorded in the final report.

