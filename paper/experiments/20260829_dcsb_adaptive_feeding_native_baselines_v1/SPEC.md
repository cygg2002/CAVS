# DCSB and Adaptive Feeding source-oriented external baselines

## Objective

Complete DCSB and Adaptive Feeding on the frozen COCO, Oregon, and VOC detector
pairs without importing CAVS actions or learners.  Every decision is binary:
retain the Edge result or upload the Original image to the frozen cloud model.

This is a same-cache reimplementation, not a claim that the authors' published
numbers are reproduced: the detector pairs and two of the three data protocols
differ from the original papers.

## Source-faithful rows

- `DCSB-Native` implements the original three-threshold rule.  A fit image is
  difficult when the cloud detector finds more confidence-0.5 boxes than the
  edge detector.  The confidence threshold matches total fit predicted boxes
  to total fit annotations.  As required by Section V-D of the paper, count and
  minimum-area threshold regression uses true fit annotations; inference uses
  only estimated values from edge detections.
- `AdaptiveFeeding-Native-w3` uses the paper's class histogram plus Top-K
  `(confidence, xmin, ymin, width, height)` proposal features and a primal
  linear SVM with `C=1`, easy/hard weights `1:3`, and threshold zero.  Hard
  labels use image-level VOC07 mAP at IoU 0.5: cloud mAPI must be strictly
  greater than edge mAPI.
- Adaptive Feeding uses Top-25 on VOC and Top-50 on COCO exactly as stated in
  the paper.  Oregon has no source setting; Top-25 is frozen before evaluation
  and reported as a cross-domain adaptation.

The original AF system used a separate Tiny-YOLO proposal generator before the
basic/partner pair.  To avoid adding a fourth, unmatched detector to the frozen
three-dataset protocol, this reimplementation takes proposals from the current
Edge detector.  The feature, mAPI label, SVM, weighting, and binary partner
choice remain source-oriented, but this proposal-source substitution is an
explicit system adaptation and prevents any claim of bitwise official
reproduction.

Native rows report their naturally achieved call and byte ratios.  They are not
forced onto the paper's 10%, 25%, or 40% work points.

## Common-budget rows

For apples-to-apples point tables, `DCSB-Margin-BudgetMatched` and
`AdaptiveFeeding-Margin-BudgetMatched` rank images using only the respective
external method's scalar margin and upload only Original JPEGs.  Fixed-call
selects the top requested image count.  Fixed-byte scans the same deterministic
ranking and accepts an image whenever its Original bytes fit.

This normalization is an evaluation adaptation and is never described as the
authors' native method.  It does not use ActionValue, SharedAction, Q90, ROI,
MAUR, or the CAVS capacity solver.

## Data isolation

- Fit annotations are the only labels used to fit either baseline.
- Calibration and final detections are used only to generate inference scores.
- Calibration and final annotations are used only by their official evaluator.
- Final scores and image bytes may be used by the frozen offline budget matcher,
  but final labels never choose a model, threshold, or method.
- Existing experiments are read-only; this package writes only below `runs/`.

## Result matrix

For each dataset and each of calibration/final:

- 2 reference rows;
- 2 native rows;
- 2 budget-matched methods x 2 modes x 3 budgets = 12 rows.

That is 16 rows per role and 32 rows per dataset.  Official AP50-95, AP50, AP75,
call ratio, byte ratio, KB/image, selected count, and capacity utilization are
recorded for every row.

## Success criteria

- Pure unit tests verify exact feature order/padding, mAPI label direction,
  DCSB threshold behavior, deterministic SVM scores, Original-only selection,
  capacity safety, and the 32-row matrix contract.
- Preflight validates role populations, source manifests, model hashes, and the
  absence of forbidden actions in selections.
- A completion marker is written only after every expected result row exists
  uniquely and all source provenance fields pass.
