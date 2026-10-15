# COCO RK3588–Cloud causal-online DCSB and Adaptive Feeding extension

## Objective

Add source-oriented DCSB and Adaptive Feeding to the already frozen COCO
RK3588-FP16 causal-online HIL protocol.  This is an additive experiment and
must not modify or overwrite any of the 46 completed final-v2 board runs.

## Frozen system

- Dataset: COCO val2017, 5,000 images in the existing images-only sequence.
- Edge detector: `yolo26n-rk3588-fp16.rknn`, SHA-256
  `0915a99235c3be5723fcfeda45b06775ecccc2908cc5b490f093f23bbdede12d`.
- Cloud detector: `yolo26x.pt`, SHA-256
  `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`.
- Roles: fit=9,678, calibration=571, final=5,000 with disjoint identities.
- Action space: Edge or Original only.  No Q90, ROI, MAUR, ActionValue, or
  SharedAction component may enter either external baseline.
- Execution: one image is opened, scored, decided, and if selected uploaded
  before the next image is opened (`zero_lookahead=true`,
  `global_plan=false`, `buffer_frames=0`).
- Budget: JPEG payload bytes at 10%, 25%, and 40%; HTTP wire bytes are reported
  separately.

## Model alignment and data isolation

- Refit both lightweight baselines from the existing RKNN-FP16 fit candidates;
  do not reuse the PyTorch-cache policy or any CAVS learner.
- DCSB uses its three-threshold rule and fit annotations only, following the
  source-oriented PC implementation.
- Adaptive Feeding uses the COCO Top-50 class-histogram plus
  `(confidence, xmin, ymin, width, height)` proposal feature and the source
  linear SVM settings (`C=1`, hard weight 3).
- Policy scoring consumes the raw post-NMS RKNN boxes exactly as the online
  runner sees them.  The committed Edge output and its AP labels separately
  remove non-finite or degenerate boxes; these two stages must not be conflated.
- Calibration labels never fit the models.  Calibration margins may only
  freeze a monotone empirical score normalization and byte-controller prices.
- Final labels, final AP, and future final frames may not train, normalize,
  tune, or select any policy or control.

## Formal board matrix

The extension has eight new, serial hardware runs:

| Method | Work point | Runs |
|---|---|---:|
| DCSB-Margin-BudgetMatched | 10% / 25% / 40% | 3 |
| AdaptiveFeeding-Margin-BudgetMatched | 10% / 25% / 40% | 3 |
| DCSB-Native | source threshold, natural cost | 1 |
| AdaptiveFeeding-Native-w3 | source threshold, natural cost | 1 |

The six budget-matched rows are the direct comparison to CAVS-OQR.  The two
native rows preserve the external papers' natural decision thresholds and are
reported separately because their byte ratios are not 10%/25%/40% work points.

## Acceptance gates

Every formal run must have 5,000 committed frames, zero cloud failures, zero
prefix-budget violations, no invalid final boxes, wire/payload ratio below
1.02, and hash-bound model, policy, control, sequence, RKNN, and cloud assets.
The board package must pass local and board tests plus a fresh 50-frame smoke
for both method families before formal execution.
