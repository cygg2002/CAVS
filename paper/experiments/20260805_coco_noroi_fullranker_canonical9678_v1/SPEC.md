# COCO Canonical No-ROI Full-Ranker Comparison

## Question

On the canonical COCO split, does the frozen DualHead ranker outperform the
external ORIC/MORIC baselines and the nine historical internal rankers when ROI
is removed from the action space?

## Frozen population

- Ranker fit: 9,678 train2017 images.
- Calibration: 571 disjoint train2017 images.
- Evaluation: the already accessed 5,000-image val2017 population.
- Budgets: 10%, 25%, and 40%.
- Fixed-byte is primary; Fixed-call is secondary.
- The result is a post-hoc completeness extension, not a new hidden final.

## Methods

The comparison contains the nine historical internal rankers, Current-BudgetSet,
SetValue, DualHead, and four canonical ORIC/MORIC variants. All checkpoints are
frozen and reused; no ranker is fit on calibration or val2017.

## Payload policies

1. `Original-only`: selected images use Original; unselected images use Edge.
2. `SharedAction-NoROI`: selected images use either Original or Q90. The frozen
   SharedAction Q90 score and threshold are retained and ROI is hard-disabled by
   setting its threshold to positive infinity.

`CostEligible-AllROI` and `GBDT-Hierarchical` are intentionally excluded because
their defining operation is an ROI substitution. With ROI disabled they are not
distinct policies.

## Selection and evaluation

- Fixed-call selects exactly round(N * budget) images by frozen ranker score.
- Fixed-byte capacity equals budget times the all-Original val2017 bytes.
- Fixed-byte lambda is selected on calibration only, independently for each
  method, policy, and budget.
- Official AP is COCO AP50-95 at confidence 0.001.
- Every result row must have ROI ratio zero.

## Statistical comparisons

Run 1,000 paired image bootstraps for DualHead against, per policy and budget:

- the strongest external ORIC/MORIC variant frozen on calibration;
- the strongest historical internal ranker frozen on calibration;
- SetValue.

## Success criteria

- Exactly 192 unique rows for each evaluated role: 16 methods x 2 policies x
  2 modes x 3 budgets.
- No ROI action appears in any selection or result.
- Every Fixed-byte selection respects its capacity.
- Frozen score IDs exactly match canonical action-cache IDs.
- val2017 is never used for training, lambda selection, or comparator selection.

