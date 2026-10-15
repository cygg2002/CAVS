# COCO DualHead val2017 Paper Evaluation v1

## Frozen contract

- Learner checkpoint: trained only from the completed 5,000-image COCO
  train2017 development role.
- Test population: all 5,000 COCO val2017 images from the canonical paper
  cache; no subdivision and no train2017 image is evaluated.
- Candidate: DualHead. Registered ablations: Current-BudgetSet and SetValue.
- External controls: ORIC/MORIC AP50 and AP50-95 variants trained on the same
  train2017 development role and stored in the frozen checkpoint.
- Primary protocol: exact Original-byte capacity at 10%, 25%, and 40%.
- Secondary protocol: exact fixed-call 10%, 25%, and 40%.
- The strongest external comparator is frozen from development OOF before
  val2017 access: ORIC-AP5095 at all three fixed-byte work points.
- Report all point estimates and 1,000 deterministic paired image bootstraps.
  val2017 does not select a model, epoch, threshold, or comparator.

## Canonical data

The test samples, ground truth, Edge predictions, Cloud Original predictions,
and exact JPEG bytes are reused read-only from the completed canonical COCO
paper protocol. Their pre-existing hashes are pinned in `configs/protocol.json`.

## Success criteria

- Exactly 5,000 unique val2017 images and zero train2017 paths in evaluation.
- Frozen checkpoint and all four canonical sources match registered hashes.
- Seven methods x two modes x three budgets are reported.
- Fixed-byte selections never exceed capacity.
- Final output records 1,000 paired bootstrap replicates at every budget.

