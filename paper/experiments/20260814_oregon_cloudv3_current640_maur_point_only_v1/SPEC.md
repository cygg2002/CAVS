# Oregon Cloud-V3 Current@640+MAUR point-only specification

## Goal

Evaluate the already completed strict three-seed full-fit `CAVS-OQR-MAUR`
learner on the frozen Oregon final split at the three paper Fixed-byte budgets.

## Frozen inputs

- Source experiment: `20260813_coco_oregon_cavs_final_completion_v1`
- Dataset: Oregon canonical event-disjoint split
- Cloud model: Cloud-V3 epoch-28, SHA-256
  `d3890a57fc7d567880a5b01b7eb949387d9b7058ce7ee25bdb9471d9bb4642d9`
- Learner seeds: `20260810`, `20260811`, `20260812`
- Action space: Edge / Original / Q90 / Current@640
- ROI fusion and reward semantics: MAUR
- Budgets: Fixed-byte 10%, 25%, 40%

## Boundaries

- Do not train or refit any learner.
- Do not run bootstrap.
- Do not write into any source experiment.
- Do not use final data for training, calibration, or method selection.
- Report only the frozen three-seed ensemble point estimates.

## Success criteria

- Exactly three result rows are produced.
- Every selection respects its byte capacity and uses at least 98% of it.
- All three frozen full-fit score files align exactly with the final records.
- Source experiments remain read-only.

