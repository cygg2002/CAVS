# Spec: COCO RKNN-FP16 Causal Online HIL

## Objective

Build a zero-lookahead image-stream replay on Orange Pi 5 Ultra / RK3588.
For every frame, the board must finish Edge inference, method scoring, byte
admission, any real HTTP cloud request, and fusion before it reads the next
image.  The completed two-pass fixed-byte HIL remains an offline global upper
bound and is never modified by this experiment.

## Frozen System

- Dataset: COCO val2017, 5,000 images in one hash-pinned images-only order.
- Edge: RKNN FP16 model SHA-256
  `0915a99235c3be5723fcfeda45b06775ecccc2908cc5b490f093f23bbdede12d`.
- Cloud: `yolo26x.pt` SHA-256
  `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`.
- Budgets: 10%, 25%, and 40% of the cumulative native-Original JPEG bytes.
- CAVS ROI: Current@640 with MAUR fusion.
- Model selection and byte-price calibration: fit/calibration only.

## Causal Contract

At frame `t`, earned byte credit is

`rho * sum_{j<=t}(OriginalBytes_j)`.

The controller may select action `a` only when

`sum_{j<t}(SelectedBytes_j) + SelectedBytes(t,a) <= earned credit`.

For CAVS, among feasible actions it maximizes

`Value(t,a) - lambda * SelectedBytes(t,a) / median_original_bytes`.

`lambda` and the median are frozen on calibration.  The online process may not
read final candidates, final scores, COCO ground truth, future image bytes, or
any global `plans.json`.

## Paper Matrix

Primary Fixed-byte methods at all three budgets:

1. CAVS-OQR (primary system).
2. CAVS-OQ (ROI ablation; independently trained).
3. CAVS-O (action-matched learned baseline; independently trained).
4. ORIC and MORIC (historical frozen source-style single-model baselines).
5. Confidence (frozen `1 - min(conf >= 0.25)`, empty set = 2.0).
6. Random-50 (SHA-256 image score; one representative hardware seed).

The available ORIC/MORIC weights are not three-seed RKNN-retrained ensembles
and must be labelled `score_adapted_historical_policy`. Their byte-aware causal
controller is the unified-system adaptation, not the original authors' native
threshold controller. AP50/AP50-95 metric-specific variants may be added only
after separately trained weights exist; one model must never be relabelled as
both variants.

CAVS-O and CAVS-OQ must be independently trained on their own action spaces.
Masking the OQR output at runtime is allowed only as a diagnostic and is not a
paper-level action-space ablation.

## Project Structure

- `board_bundle/`: standalone causal runner and immutable runtime assets.
- `host/`: fit/calibration-only policy training, control calibration, bundle
  preparation, and post-run COCO evaluation.
- `mac_server/`: hash-pinned authenticated cloud endpoint.
- `runs/`: generated artifacts only.

## Testing Strategy

- Unit: exact rational byte accounting, deterministic ties, prefix invariance,
  suffix independence, state restore, policy hashes, and method scores.
- Integration: a fake Edge runtime and fake cloud verify strict frame ordering,
  no annotation access, failure fallback, and resume equivalence.
- Hardware smoke: 50 images with a fresh run ID and no failed request.
- Formal: one independent 5,000-image run per method and budget.

## Boundaries

- Always: use actual payload bytes; bind run contracts to sequence, controls,
  policy, RKNN, RKNN preprocessing/decode helpers, runtime bundle, and cloud
  hashes; require a runtime bearer token.
- Ask first: changing action semantics, cloud model, image order, or budget rule.
- Never: modify the completed global HIL, select a controller on final, reuse
  historical PyTorch ORIC/MORIC policies as formal RKNN baselines, or call the
  system real-time before deadline measurements support that claim.

## Success Criteria

1. Maximum prefix budget violation is exactly zero.
2. Identical input prefixes produce bit-identical actions and controller state.
3. Removing COCO annotations does not affect online execution.
4. Each frame completes before the next frame image is opened.
5. An interrupted run resumed at a committed frame matches an uninterrupted run.
6. Formal runs report AP50-95/AP50/AP75, byte utilization, action counts,
   failures, FPS, and P50/P95/P99 timing for every stage.
