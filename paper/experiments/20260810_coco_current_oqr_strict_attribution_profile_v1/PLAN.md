# Implementation Plan

## Phase 1: contracts

- [x] Add protocol and pure helper tests.
  - Verify: tests initially fail because `scripts/core.py` is absent.
- [x] Implement protocol validation, deterministic Original-only selection,
      ensemble averaging, parameter counting, latency summaries, and analytic
      Linear FLOPs.
  - Verify: all unit tests pass.

## Phase 2: frozen evaluation

- [x] Preflight source counts, IDs, checkpoints, and frozen result artifacts.
- [x] Re-evaluate Current-OQ/OQR for each seed and ensemble under Fixed-byte.
- [x] Re-evaluate their Original-action score under Original-only Fixed-call.
- [x] Import frozen ORIC/MORIC and Confidence rows without selection changes.
  - Verify: row counts and exact reproduction of ensemble source AP.

## Phase 3: deployment profile

- [x] Load each Current-OQR checkpoint and verify identical architecture.
- [x] Measure one-thread model latency for batch=1 and the frozen batch size.
- [x] Measure full 5,000-image Fixed-byte scheduling latency.
- [x] Record parameters, checkpoint bytes, analytic GFLOPs, runtime environment.
  - Verify: finite timings, positive parameters, and deterministic FLOP count.

## Phase 4: report

- [x] Generate CSV/JSON artifacts and `notes/RESULTS_ZH.md`.
- [x] Run full tests and source-hash verification.

## Risks

- The source full-val run is post-hoc and cannot support a confirmatory claim.
- Current uses a three-model ensemble while frozen external methods are single
  models; compute fairness must be disclosed.
- Model-only GFLOPs exclude detectors, codecs, and network transfer.
- Current-OQR has an additional ROI action, so external comparisons are
  system-level; Current-OQ is the action-controlled comparison.
