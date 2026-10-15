# Implementation Plan: COCO Budget-Conditioned Contextual Ranker

## Architecture Decisions

- Import the frozen ORIC feature/model contracts instead of copying them.
- Keep the exact AP50 reward and top-25 feature matrix in Phase A.
- Train one shared MLP conditioned on the requested budget. Optimize pairwise
  separation across the true top-budget boundary plus a small CDF regression
  term for stable ordering.
- Average three pre-registered seeds; never select a lucky seed on calibration.
- Separate calibration and final into different commands and completion files.

## Task List

### Phase 1: Contracts

- [x] Write failing unit tests for budget pairs, loss, score shape and gate rules.
  - Acceptance: tests fail because the new core does not exist.
  - Verify: `python3 -m unittest discover -s tests -v`.
  - Files: `tests/test_ranker_core.py`, `tests/test_protocol.py`.

### Phase 2: Learner

- [x] Implement deterministic fit/validation training and ensemble inference.
  - Acceptance: synthetic signal ranks above noise at every budget.
  - Verify: all new tests pass.
  - Files: `scripts/ranker_core.py`.

### Phase 3: Frozen runner

- [x] Implement calibration-only evaluation and gated final evaluation.
  - Acceptance: calibration never loads final; final refuses a failed gate;
    all reads are hash-audited and writes stay under this experiment.
  - Verify: protocol tests and local smoke pass.
  - Files: `scripts/run_diagnostic.py`, `configs/protocol.json`.

### Phase 4: Remote execution

- [x] Run tests, calibration gate, and conditionally final on the server.
  - Acceptance: completion/report artifacts exist and reproduce all baselines.
  - Verify: source hashes unchanged and result counts complete.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Small calibration split is noisy | Pre-register three-seed ensemble and a 0.002 mean gain gate |
| Pairwise model merely memorizes reward ranks | Fit-internal holdout selects epochs; calibration is never trained on |
| AP50 target cannot improve AP50-95 | Stop Phase A cleanly, then generate AP50-95 rewards |
| Final leakage | Separate command and refuse final unless frozen gate passes |
