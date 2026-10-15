# Implementation Plan

## Phase 1: Contracts

- [ ] Add frozen protocol validation and score-composition tests.
  - Acceptance: hybrid Original equals ORIC; action differences equal Current.
  - Verify: focused unit test fails before `core.py` exists and passes after it.

## Phase 2: Point Evaluation

- [ ] Load frozen ORIC and Current ensembles with strict image-ID checks.
- [ ] Evaluate four methods under Fixed-byte and Fixed-call.
- [ ] Reproduce frozen Current and ORIC control rows.
  - Acceptance: 24 unique rows; no byte overflow; reproduction error ≤1e-9.
  - Verify: preflight and evaluation completion markers exist.

## Phase 3: Statistical Evaluation

- [ ] Run four paired comparisons per budget with shared image resamples.
- [ ] Generate a Chinese result report with point estimates and 95% CIs.
  - Acceptance: 12 comparisons ×1,000 finite replicates.
  - Verify: bootstrap and report completion markers exist.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| ORIC and Current score scales differ | Use the predeclared Original-anchor decomposition; do not tune a coefficient. |
| Image order silently differs | Require exact canonical ID equality before composing scores. |
| Existing Oregon run uses server CPUs | Point evaluation uses one low-priority process; full bootstrap waits for the active Oregon pipeline to finish. |
| Post-hoc overclaim | Mark every artifact as final reuse and report no confirmatory claim. |
