# Implementation Plan

## Phase 1: Inputs and Contracts

- [x] Freeze source paths and SHA-256 digests for calibration records, final
  frames, sequence, policy, and registered controls.
- [x] Add tests for exact-prefix and finite-burst capacity functions.

## Checkpoint 1

- [x] The fixed+prefix synthetic trace matches the reference controller.
- [x] No controller API accepts a future record or record window.

## Phase 2: State-Aware Price

- [x] Add failing tests for empirical price-demand construction and
  remaining-capacity price selection.
- [x] Implement deterministic state-aware current-frame selection.
- [x] Add suffix-invariance and final-cap tests.

## Checkpoint 2

- [x] Unit tests pass and controller overhead is measured on 5,000 records.

## Phase 3: Calibration and Final

- [x] Run the registered burst grid on 571 calibration records for each budget
  and price policy.
- [x] Freeze six controls and write all calibration trials.
- [x] Replay the four cells once on the 5,000-record final stream.
- [x] Verify fixed+prefix equality with the frozen baseline selections.

## Phase 4: Accuracy and Review

- [x] Materialize changed actions using the matching per-budget source run and
  same cloud model protocol.
- [x] Evaluate COCO AP50-95/AP50/AP75 and generate the Chinese report.
- [x] Run complete tests and five-axis code/results review.

## Risks

| Risk | Mitigation |
|---|---|
| Calibration curve overfits 571 records | No final tuning; retain fixed baseline and label diagnostic. |
| Burst improves proxy but not AP | AP is the decision gate; do not package for board on proxy alone. |
| Mixed MPS/RTX numerical backend | Use matching source per budget and disclose any changed action executed elsewhere. |
| Strong prefix and fixed-window pacing answer different constraints | Report both explicitly; never merge their byte guarantees. |
