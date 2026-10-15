# Implementation Plan: Budget-Set SharedAction + CostEligible

## Architecture Decisions

- Reuse immutable action/reward caches; do not regenerate detector outputs.
- Build a new model only inside this experiment directory.
- Keep ORIC/MORIC controls on the same training/holdout population and reward.
- Separate prepare, train and holdout commands so holdout cannot select epochs.

## Task List

### Phase 1: Contracts

- [x] Add failing tests for split isolation, set encoding, boundary pairs,
  forbidden features and resource selection.
  - Acceptance: tests fail because the new core is absent.
  - Verify: focused `unittest` command.
  - Files: `tests/test_experiment_core.py`, `tests/test_stage_contracts.py`.

### Phase 2: Pure core

- [x] Implement deployable feature construction, model, objectives and selectors.
  - Acceptance: synthetic signal and permutation tests pass.
  - Verify: all new unit tests pass.
  - Files: `scripts/experiment_core.py`.

### Phase 3: Isolated runner

- [x] Implement prepare/train/holdout stages and immutable artifacts.
  - Acceptance: train refuses holdout IDs/rewards; holdout runs once only after
    completed checkpoints exist.
  - Verify: stage-contract tests and a 32-image smoke run.
  - Files: `scripts/run_experiment.py`, `configs/protocol.json`.

### Phase 4: Full execution

- [x] Run full training and one sealed-holdout evaluation on the server.
  - Acceptance: result matrix is complete for four methods, two modes and three
    budgets; source fingerprints remain unchanged.
  - Verify: completion JSON, CSV and report hashes.

### Phase 5: Review

- [x] Review code, run regression tests and write the Chinese result note.
  - Acceptance: no high-severity review issue and conclusions follow frozen gates.
  - Verify: review checklist and result-note cross-check.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Reward contexts span all 5,000 images | Not a strict independent final | Label run transductive and use direct holdout AP only as diagnostic |
| Multi-action gain is mostly routing, not action choice | Innovation attribution unclear | Report `BudgetSet-Full` and `BudgetSet-SharedAction` separately |
| Old GAIR fields leak teacher signals | Inflated learned result | Whitelist deployable fields; test forbidden fields have no effect |
| Byte and fixed-call protocols diverge | Unfair ORIC comparison | Report both modes and use identical records/actions/capacity |
| Holdout overfitting | Invalid evidence | One evaluation only; no post-holdout tuning |
