# COCO Action-Conditioned Value–Risk OOF Diagnostic v1

## Objective

Test whether learning a separate value and lower-confidence reward for every
`(image, Original/Q90/ROI)` pair improves the current Original-only DualHead +
frozen SharedAction pipeline.  This is a cache-only go/no-go diagnostic, not a
paper final.

## Frozen data contract

- Training diagnostic: canonical COCO train2017 `fit`, exactly 9,678 images.
- Generalization check: canonical train2017 `calibration`, exactly 571 images.
- Actions: Edge, Original, Q90, and the existing canonical ROI/MAUR action.
- Detector predictions, payload bytes, availability, GT and ROI geometry remain
  read-only.
- Contextual AP50-95 rewards are the already completed static-context caches in
  `20260805_coco_selection_context_jointaction_teacher_v1`.
- Audit, val2017/final and every registered final are forbidden.

## Compared methods

1. `CurrentDualHead-SharedAction`
   - reproduce the current mechanism with an Original-only DualHead image value;
   - use the frozen canonical SharedAction map to assign Original/Q90/ROI.
2. `ActionValue-Joint`
   - train one cost-free mean reward for every available action;
   - jointly select image and action under call/byte budgets.
3. `ActionValueRiskLCB-Joint`
   - share the same set/action encoder;
   - predict mean reward and a 0.10 conditional reward quantile;
   - schedule with the lower quantile as a conservative value.

All three methods use the same top-64 permutation-invariant Edge detection set,
the same optimizer, fold assignment and one diagnostic seed.  Q90 receives its
own action token; byte cost is exact metadata and is never predicted.

## Evaluation

- Five deterministic OOF folds over fit=9,678.
- One full-fit training followed by a single frozen calibration=571 check.
- Fixed-call budgets select exactly 10%/25%/40% images and at most one action
  per image.
- Fixed-byte budgets use the same capacity: budget times all-Original bytes.
- Report Official AP50-95/AP50/AP75, KB/image, action mix, selected harmful ROI
  rate, Teacher regret, and capacity checks.
- Precomputed contextual reward context may cross OOF folds; this limitation is
  disclosed and prevents treating the OOF result as a paper estimate.

## Success gate

The candidate passes a role only when, relative to
`CurrentDualHead-SharedAction`:

- at least two Fixed-byte budgets improve Official AP50-95 by at least 0.001;
- the remaining Fixed-byte budget is no worse than -0.001;
- every Fixed-call budget is no worse than -0.001;
- all capacities and strict call counts pass.

The experiment opens a three-seed paper protocol only if
`ActionValueRiskLCB-Joint` passes both OOF and calibration.  Otherwise the
static action-conditioned learner branch stops.

## Commands

```bash
python -m unittest discover \
  -s paper/experiments/20260805_coco_actionconditioned_valuerisk_oof_v1/tests -v
python paper/experiments/20260805_coco_actionconditioned_valuerisk_oof_v1/scripts/run_oof.py --stage all
```

## Boundaries

- Always: verify hashes, exact role counts, OOF exclusivity, capacities, and
  completion artifacts.
- Ask first: change actions, folds, model seed, quantile, budgets, or gate.
- Never: retrain detectors, alter source experiments, access audit/val2017/final,
  or tune on calibration after its first evaluation.

