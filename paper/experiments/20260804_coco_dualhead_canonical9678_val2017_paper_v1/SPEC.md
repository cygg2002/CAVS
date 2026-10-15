# COCO canonical 9,678 paper comparison

## Objective

Retrain Current-BudgetSet, SetValue, DualHead, ORIC and MORIC under the same
canonical COCO population used by the existing paper experiments, then compare
them on the complete COCO val2017 set.

## Frozen data contract

- Training: canonical `train2017/fit`, exactly 9,678 images.
- Comparator calibration: canonical `train2017/calibration`, exactly 571 images.
- Evaluation: canonical `val2017/final`, exactly 5,000 images.
- The 630-image canonical action audit is not used because this experiment has
  only Edge and Original actions and does not calibrate ContextM2E/ROI action
  thresholds.
- No checkpoint trained on the temporary 5,000-image development split may be
  loaded or reused.

## Frozen method contract

- Methods: ORIC-AP50, MORIC-AP50, ORIC-AP5095, MORIC-AP5095,
  Current-BudgetSet, SetValue and DualHead.
- Actions: Edge or Cloud Original only. No Q90 or ROI action is added.
- Budgets: 10%, 25% and 40%.
- Fixed-byte is the primary mode; Fixed-call is secondary.
- Model architecture, loss weights, optimizer settings, seeds and contextual
  reward configuration are copied unchanged from the prior DualHead experiment.
- Contextual rewards are regenerated using only fit=9,678 predictions and ground
  truth. Calibration and val image IDs are forbidden from reward construction.

## Evaluation contract

- The strongest external ORIC/MORIC comparator at each fixed-byte budget is
  frozen on calibration=571.
- The frozen models and comparators are evaluated once by this experiment on
  complete val2017=5,000.
- Report Official AP50-95, AP50, AP75, KB/image, upload ratio, GFLOPs/image and
  E2E@10Mbps for every method, budget and mode.
- Run 1,000 paired bootstrap replicates for DualHead versus each frozen external
  comparator on the primary fixed-byte work points.

## Success criteria

- All population and split assertions pass before reward generation or training.
- All seven methods share identical fit/calibration/val populations and detector
  caches.
- Final comparison contains 7 methods x 2 modes x 3 budgets = 42 rows.
- No result from either temporary 5,000-train experiment is merged into the main
  report.

