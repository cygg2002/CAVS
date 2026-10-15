# Implementation Plan

## Architecture Decisions

- Derive a standalone bundle from the frozen 20260817 online package; do not
  edit the source package.
- Extend the controller state only with current price, step size, and bounds.
- Preserve the existing sequential stream engine so immediate upload ordering
  remains structural rather than a command-line convention.
- Tune only causal-control hyperparameters; the CAVS learner stays frozen.

## Phase 1: Causal Controller

- [x] Add failing tests for update direction, bounds, exact prefix feasibility,
  fixed-price parity, deterministic resume, and frame event ordering.
- [x] Implement the smallest projected primal-dual controller satisfying them.
- [x] Benchmark controller-only decision overhead.

## Checkpoint: Controller

- [x] All board unit tests pass under Python 3.12 and the host Python.
- [x] No test or runtime API accepts a future frame or a frame window.

## Phase 2: Replay and Calibration

- [x] Build hash-bound causal records from the existing RKNN-FP16 policy and
  deployment candidates.
- [x] Search the registered step-size grid on fit/calibration only.
- [x] Freeze one control per budget and compare it with `eta=0` on identical
  order and candidate records.

## Checkpoint: Frozen Controls

- [x] Three controls validate and state `final_used_for_tuning=false`.
- [x] Every replay has zero prefix violation and a complete decision trace.

## Phase 3: Final and Board Bundle

- [x] Evaluate final once with frozen controls and report AP where all selected
  action outputs are available.
- [x] Package policy, controls, runtime, sequence, and manifests for RK3588.
- [x] Run bundle verification and a fake-cloud immediate-upload integration test.

## Checkpoint: Complete

- [x] Existing frozen experiment remains byte-for-byte unchanged.
- [x] The new tarball and SHA-256 are ready for a 50-frame board smoke followed
  by 5,000-frame formal runs.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Adaptive price oscillates on variable JPEG sizes | Normalize by the frozen median and tune a small registered grid. |
| Calibration proxy improves but AP does not | Keep AP as the final gate and retain the fixed controller if needed. |
| Missing counterfactual cloud outputs locally | Report proxy replay separately and require real board HTTP for final AP. |
| Resume changes price history | Recompute state from every committed observation and compare snapshots. |
