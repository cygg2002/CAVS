# Implementation plan

## Architecture decisions

- Build one new isolated experiment and import completed implementations rather
  than copying model code.
- One orchestrator launches one resumable process per dataset. Each process uses
  multiprocessing for cached detection statistics/rewards and bounded parallel
  training jobs.
- Train each dataset once. Track A and Track B consume the same frozen
  three-seed checkpoints.
- Treat Track B as an append-only post-hoc extension; do not rerun the nine
  historical rankers or external ORIC/MORIC.

## Tasks

### Phase 1 — contracts

- [ ] Add frozen protocol and pure validation helpers.
  - Acceptance: dataset roots/counts, methods, policies, budgets, thresholds,
    result status, and expected row counts are machine checked.
  - Verify: protocol tests fail before implementation and pass after it.

- [ ] Add score/action/capacity contract helpers.
  - Acceptance: exact score alignment, no forbidden action, identical Track B
    membership, and capacity overflow are detected.
  - Verify: focused unit tests cover every failure mode.

### Phase 2 — reusable pipeline

- [ ] Add preflight and cached contextual-reward stages.
  - Acceptance: canonical sources are hashed; fit/calibration only are touched;
    stages are resumable.
  - Verify: remote preflight plus subset smoke test.

- [ ] Add three-seed model training and 5-fold OOF scoring.
  - Acceptance: four Track A methods produce finite, image-aligned scores; all
    training is CPU-only and fit-only.
  - Verify: tiny synthetic/unit test and real subset smoke test.

- [ ] Add Track A evaluator.
  - Acceptance: 48 unique rows, safe capacities, fixed-call counts, and all
    required metrics.
  - Verify: matrix validator and smoke output.

- [ ] Add Track B frozen-policy evaluator and bootstrap.
  - Acceptance: 18 unique rows, exact policy membership, frozen thresholds,
    post-hoc labels, and 200-replicate comparisons.
  - Verify: source-table reproduction checks and matrix validator.

### Phase 3 — launch

- [ ] Sync only the new experiment to the remote repository.
- [ ] Run tests and preflight remotely.
- [ ] Launch ENA24 and Oregon concurrently with `nohup`, 24 cache workers per
  dataset and bounded training workers.
- [ ] Save PIDs, command, log paths, and launch state; perform one immediate
  health check, then stop monitoring.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| ENA24 image IDs repeat across roles | Role-local order and population checks; never join roles by ID |
| Oregon Cloud-V2 contamination | Whitelisted canonical root and forbidden-path validation |
| Frozen policy drift | Reuse saved GBDT model/threshold; do not recalibrate |
| CPU oversubscription | Two dataset processes, bounded worker pools, one Torch thread per training worker |
| Partial long run | Atomic artifacts and completion markers per stage/dataset |
| Post-hoc result overclaim | Result-status fields and generated-note disclaimer |

