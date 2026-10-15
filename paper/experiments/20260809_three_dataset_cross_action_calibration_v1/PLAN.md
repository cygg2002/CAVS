# Implementation Plan

## Phase 1: Frozen contracts

- [ ] Add `protocol.json` with the exact 3×2×2×3×5 matrix.
- [ ] Add failing tests for protocol, PAVA, cross-fitting, Oracle membership,
  capacity, and result-row completeness.

Checkpoint: tests fail only because the new implementation is absent.

## Phase 2: Pure calibration slice

- [ ] Implement binned PAVA fit/predict.
- [ ] Implement role-order target extraction and cross-fitted calibration.
- [ ] Implement exact result-matrix and gate validation.

Checkpoint: pure unit tests pass without loading experiment caches.

## Phase 3: Scheduling/evaluation slice

- [ ] Reuse frozen source loaders, rewards, schedulers, and evaluators.
- [ ] Implement the symmetric membership/action Oracle decomposition.
- [ ] Persist calibrators, selections, metrics, diagnostics, and completion
  markers atomically.

Checkpoint: synthetic integration tests and local test suite pass.

## Phase 4: Parallel execution

- [ ] Sync only the new isolated experiment directory to the server.
- [ ] Run three independent preflights.
- [ ] Launch COCO, ENA24, and Oregon concurrently on CPU with resumable outputs.
- [ ] Persist PIDs, per-dataset logs, and pipeline status.

Checkpoint: background jobs are alive or already complete; no GPU/final access.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| OOF calibration leakage | Second cross-fitting layer with tested train/held-out disjointness |
| ENA/Oregon image-ID reuse | Never join roles; verify role-local identities |
| Oracle silently changes membership | Fixed-membership selector asserts exact ID equality |
| Capacity violation | Reuse canonical scheduler and assert bytes after every selection |
| Accidental final/Cloud-V2 read | Source allowlist, forbidden substrings, and preflight hashes |
| Overclaiming | Oracle rows carry explicit GT-aware/non-deployable flags |
