# Implementation Plan

## Phase 1: Contracts

- [ ] Add protocol and pure validators.
- [ ] Add failing tests for the four-action space, job count, and result matrix.

## Phase 2: Training

- [ ] Load Cloud-V3 common/reward artifacts read-only.
- [ ] Retrain CurrentActionValue on O/Q/current@640 with 5-fold event OOF and
      three full-fit seeds.
- [ ] Verify checkpoint provenance and score alignment.

## Phase 3: Evaluation

- [ ] Ensemble the three full-fit models.
- [ ] Evaluate Fixed-byte and Fixed-call 10/25/40 on calibration and final.
- [ ] Join the frozen OQ/OQR12 source rows without changing their policies.

## Phase 4: Statistics and Report

- [ ] Run paired event/image bootstrap for OQR vs OQ and OQR12.
- [ ] Generate machine-readable results and `RESULTS_ZH.md`.
- [ ] Launch with limited low-priority CPU workers.

## Risks

- Numerical image IDs repeat across roles: keep every role namespaced and verify
  the role-local sample order.
- The source experiment is still active: all writes are restricted to this new
  directory and workers are CPU-limited.
