# Implementation Plan: Selection-Context JointAction Teacher

## Architecture decisions

- Reuse the audited ORIC `DetectionStats` and AP50-95 reward implementation.
- Reuse the audited GAIR multiple-choice Lagrangian scheduler.
- Keep the first experiment one-step and fixed-byte only; no new model or GPU
  inference is in scope.
- Store expensive per-image stats and reward matrices as resumable artifacts.

## Tasks

### Phase 1: pure contracts

- [x] Add protocol validation, deterministic cross-role context sampling,
  candidate-group construction, action scheduling, capacity checks, and gate
  evaluation.
  - Acceptance: one action per image; unavailable actions excluded; edge has
    zero cost/value; gate implements the frozen thresholds.
  - Verify: focused unit tests fail before implementation and pass afterward.

### Phase 2: cache pipeline

- [x] Add read-only preflight and source fingerprinting.
  - Acceptance: exact fit/calibration counts; disjoint IDs; four action caches;
    final/audit never loaded.
  - Verify: preflight report binds every source path and hash.
- [x] Add resumable DetectionStats and contextual-reward generation.
  - Acceptance: IoU 0.50:0.95, confidence 0.001, deterministic context size 32;
    fit static rewards and calibration static/contextual rewards are complete.
  - Verify: smoke stage produces finite matrices with correct masks/shapes.

### Phase 3: evaluation and launch

- [x] Schedule both teachers, evaluate Official calibration metrics, write the
  result table, gate, and Chinese summary.
  - Acceptance: six primary rows, exact capacities, action mixtures and overlap,
    no source drift.
  - Verify: local and remote tests pass; background run reaches a checkpoint or
    completion without touching source experiments.

## Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Contextual rewards are CPU-heavy | Medium | fork workers, stage checkpoints, smoke benchmark |
| Oracle reward differs from Official COCO AP | Medium | use ORIC only for selection; gate on Official evaluator |
| Selection context leaks calibration state | High | context images and selected actions come from fit only |
| One-step context understates iterative ceiling | Low | treat failure as evidence against this simple branch, not a universal impossibility proof |

## Outcome

- Completed on 2026-08-05 with 8/8 unit tests and artifact-contract checks passing.
- The fixed-byte gate failed: AP deltas at 10%/25%/40% were
  `-0.002629/+0.001683/-0.002375`.
- No audit, val2017/final, or registered-final role was accessed.
- Decision: do not implement the full Selection-Context learner from this
  one-step teacher; retain the existing SharedAction, CostEligible-AllROI, and
  GBDT-Hierarchical action frameworks.
