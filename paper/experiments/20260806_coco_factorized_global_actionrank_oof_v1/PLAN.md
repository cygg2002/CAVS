# Plan: COCO Factorized Global Action Rank OOF Diagnostic v1

## Architecture decisions

- Reuse the completed relational experiment as an immutable source of
  fold-contained rewards and three baseline OOF score tensors.
- Add one model only: a shared mean/max image encoder with separate global
  image-value and cloud-action-advantage paths.
- Keep cost outside the learner and use the same exact schedulers/evaluator.
- Fail closed: OOF gate precedes calibration; forbidden roles remain sealed.

## Tasks

### Task 1: Freeze contracts and provenance

- Acceptance: exact data roles, methods, actions, source artifacts, hashes,
  folds, seeds, budgets, losses, and gate are machine validated.
- Verify: protocol tests and remote preflight.
- Files: `SPEC.md`, `PLAN.md`, `configs/protocol.json`.

### Task 2: Specify factorized behavior with failing tests

- Acceptance: tests require correct global/advantage targets, reconstruction,
  action-difference independence from global value, and finite gradients.
- Verify: tests fail before the factorized functions/model exist.
- Files: `tests/test_core.py`.

### Task 3: Implement the factorized learner

- Acceptance: the model and four-part objective satisfy Task 2 without
  changing the frozen baseline implementation.
- Verify: all unit tests pass.
- Files: `scripts/core.py`.

### Checkpoint A

- Tests pass locally.
- No source experiment has changed.
- No payload feature reaches the learner.

### Task 4: Implement cache-reuse OOF and evaluation

- Acceptance: 15 new jobs complete; three baseline score tensors are hash- and
  ID-verified; the assembled 96-row matrix is No-ROI and within capacity.
- Verify: preflight, score matrix assertions, and completion markers.
- Files: `scripts/run_experiment.py`.

### Task 5: Conditional calibration and report

- Acceptance: calibration is skipped after OOF failure or evaluated exactly
  once after OOF success; report includes AP/KB, action counts, seed support,
  and forbidden-role flags.
- Verify: gate branch tests and final completion metadata.

### Task 6: Remote launch

- Acceptance: remote tests and preflight pass, a resumable background PID/log
  exists, and the process uses CPU workers only.
- Verify: one immediate process/log check, then stop monitoring.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Reused scores differ from the current IDs | Hash and exact ordered-ID checks |
| Factorized scales are not identifiable | Define global target as best Edge/cloud reward and regress relative advantages |
| Calibration becomes tuning | OOF-only gate; one-time frozen calibration branch |
| Repeating old work | Reuse reward and three baseline score artifacts read-only |
