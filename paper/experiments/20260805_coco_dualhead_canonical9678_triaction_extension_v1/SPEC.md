# COCO canonical DualHead TriAction extension

## Objective

Evaluate the seven rankers trained by the frozen canonical fit=9,678 experiment
inside the existing paper-level COCO action framework.  This isolates whether a
better image ranker remains useful after Q90 and ROI payload decisions are
enabled.

## Frozen sources

- Ranker checkpoint: read-only output of
  `20260804_coco_dualhead_canonical9678_val2017_paper_v1`.
- Action data: canonical COCO Original/Q90/ROI caches from
  `20260720_frozen_detector_multiranker_contextm2e`.
- SharedAction head and gate: frozen canonical benchmark artifacts.
- CostEligible-AllROI and GBDT-Hierarchical: exact action construction used by
  `20260729_three_dataset_oric_moric_triaction_extension`.
- GBDT checkpoint and shared thresholds: frozen outputs from
  `20260723_three_dataset_nineranker_triaction_paper_exploration`.

No ranker, action head, ROI gate, threshold, or detector is retrained here.
Every source file is hashed before and after evaluation.

The extension may run concurrently with the source experiment's paired
bootstrap once its training checkpoint, point report, and exact selections are
frozen.  Source bootstrap output is neither read nor required by this action
extension.

The source experiment patched only its canonical runner/bootstrap core after
training to repair multiprocessing serialization.  A restricted checkpoint
load therefore permits fingerprint drift only for `canonical_runner` and
`canonical_core`; the unchanged legacy scoring implementation, reward code,
protocol, detector weights, and all role manifests must still match the
training-time fingerprint.  As an additional guard, all 42 frozen
Original-only calibration/final selections must be reproduced exactly before
the exported scores may be used.

## Population and matrix

- Ranker training provenance: COCO train2017 fit=9,678.
- Action/lambda calibration: COCO train2017 calibration=571.
- Evaluation: complete COCO val2017=5,000.
- Rankers: ORIC-AP50, MORIC-AP50, ORIC-AP5095, MORIC-AP5095,
  Current-BudgetSet, SetValue, DualHead.
- Payload policies: SharedAction, CostEligible-AllROI, GBDT-Hierarchical.
- Modes: Fixed-call and Fixed-byte.
- Budgets: 10%, 25%, and 40% for both modes.
- Expected final matrix: 7 x 3 x 2 x 3 = 126 rows.

Fixed-call uses identical selected-image membership across all three payload
policies for a given ranker and budget.  Fixed-byte recalibrates only the
existing Lagrangian byte multiplier on calibration=571 for each frozen
ranker/policy/budget; it does not tune a learned model.

## Statistical comparison

For each payload policy and Fixed-byte budget, the strongest external
ORIC/MORIC comparator is frozen on calibration.  DualHead is compared with that
comparator on val2017 using 1,000 paired image bootstraps.  Point estimates are
written before bootstrap so the experiment remains resumable.

## Result status

This is a post-hoc canonical-val2017 action extension, not a new hidden final:
the same val2017 population was already accessed by the source ranker
experiment.  It is suitable for a controlled action-framework comparison, but
an eventual confirmatory claim requires a newly sealed evaluation population.

## Success criteria

- Source ranker experiment finishes without being changed by this experiment.
- Frozen checkpoint provenance states fit=9,678 and no calibration/final
  training.
- All canonical roles and Original/Q90/ROI cache hashes pass.
- The 126-row result matrix is complete and duplicate-free.
- Fixed-call membership is identical across payload policies.
- No source hash changes between preflight and completion.
