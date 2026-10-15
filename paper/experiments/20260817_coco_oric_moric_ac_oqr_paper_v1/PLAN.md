# Implementation Plan

## Phase 1: Contracts

- [ ] Freeze source paths, source hashes, role counts and method names.
- [ ] Add tests for the 36-job matrix, fold-local unified CDF and forbidden
  deployment-domain inputs.

Checkpoint: contract tests fail before implementation and pass afterward.

## Phase 2: Methods

- [ ] Implement the Top-25 image encoder plus the frozen cost-free CAVS action
  encoder and joint value head.
- [ ] Implement ORIC raw-reward and MORIC fold-local CDF training with the same
  inner split, loss, optimizer and early stopping.
- [ ] Save fold/full artifacts and aligned score matrices.

Checkpoint: synthetic fit/predict tests pass and cost-only perturbations do not
change model inputs.

## Phase 3: Paper Evaluation

- [ ] Evaluate each seed and the ensemble under Fixed-byte and Fixed-call.
- [ ] Import only hash-verified CAVS and scalar SharedAction selections.
- [ ] Run the five registered contrasts at three budgets with paired bootstrap.
- [ ] Generate a Chinese results report with point estimates, mean/SD and CIs.

Checkpoint: 36/36 training jobs, complete point matrix and 15/15 bootstrap jobs.

## Phase 4: Remote Launch

- [ ] Run local unit tests and remote preflight.
- [ ] Sync only the isolated experiment directory.
- [ ] Launch with `nice=19`, eight workers and resumable completion markers.

Checkpoint: PID and log are recorded; existing server processes remain intact.

## Risks

| Risk | Mitigation |
|---|---|
| MORIC CDF leaks held-out rewards | Fit the CDF only from the inner training partition and test the reference explicitly. |
| Action costs enter the value model | Zero the established scene/action cost indices before every forward pass and test invariance. |
| Anchor protocol differs | Pin source SHA-256 values and exact CAVS AP values in preflight. |
| Server contention | Use `nice=19`, one Torch thread per worker and no more than eight workers. |
