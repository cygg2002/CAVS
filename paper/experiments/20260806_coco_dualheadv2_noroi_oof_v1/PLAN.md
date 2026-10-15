# Implementation Plan

## Phase 1: Contracts

- [ ] Add the frozen protocol and pure tests for No-ROI, score blending, alpha
  selection, and success gates.
- [ ] Implement the minimum pure core until the tests pass.

Checkpoint: unit tests pass without reading experiment caches.

## Phase 2: Frozen-source preflight and teacher gate

- [ ] Verify source completions, role populations, score alignment, and hashes.
- [ ] Reschedule the existing static/selection-context calibration rewards with
  Original/Q90 only and write the formal teacher gate.

Checkpoint: the prior one-step teacher is reproduced with ROI hard-masked and
no forbidden role accessed.

## Phase 3: DualHead-v2 OOF selection

- [ ] Reuse frozen five-fold OOF mean/lower/action-value scores.
- [ ] Schedule and evaluate ActionValue, RiskMean, and every alpha candidate in
  parallel over the 64 available server CPUs.
- [ ] Freeze one alpha per budget using fit OOF only.

Checkpoint: every schedule respects call/byte capacity and contains no ROI.

## Phase 4: Frozen calibration and report

- [ ] Evaluate ActionValue and frozen DualHead-v2 once on calibration=571.
- [ ] Apply gates, write Chinese results, source hashes, selections, and a
  resumable completion marker.
- [ ] Run all tests and launch the pipeline with `nohup`.

Checkpoint: PID/log/status exist; the process can finish without an attached
Codex session.

## Risks and mitigations

- COCO metric evaluation is CPU-heavy: parallelize independent points with a
  bounded fork pool and keep OpenMP threads at one per worker.
- Lower-quantile scores previously collapsed utilization: reject alpha points
  below 98% capacity rather than silently calling them safe.
- The cached contextual reward may cross OOF folds: disclose this and keep the
  result diagnostic-only.
- Existing source experiments may be edited concurrently: hash them at
  preflight and verify again before completion.

