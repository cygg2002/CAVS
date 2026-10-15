# COCO Selection-Context JointAction Teacher Diagnostic v1

## Objective

Test whether conditioning contextual AP50-95 action rewards on an already
selected action set improves the frozen canonical COCO `Edge/Original/Q90/ROI`
teacher ceiling.  This is a go/no-go diagnostic for Selection-Contextual
SharedAction; it does not train a deployable scheduler.

## Frozen data contract

- Context/reference role: canonical COCO train2017 `fit`, exactly 9,678 images.
- Held-out diagnostic role: canonical COCO train2017 `calibration`, exactly 571
  images.
- Candidate actions: `edge`, `original`, `q90`, and the existing canonical
  fixed ROI/MAUR cache.
- COCO val2017/final, canonical audit, and every registered final are forbidden.
- Detector predictions, GT, payload bytes, ROI geometry, and availability are
  read only from
  `20260720_frozen_detector_multiranker_contextm2e/runs/canonical_v1/common/coco`.

Every source is hashed during preflight and checked again before completion.

## Teacher definitions

The contextual reward is the paper-faithful ORIC construction evaluated over
IoU 0.50:0.95.  Each target uses 32 deterministic fit images as context.

1. `Static-JointAction`:
   - every context image uses its Edge prediction;
   - each target receives separate Original/Q90/ROI rewards;
   - the existing capacity-safe multiple-choice Lagrangian scheduler chooses
     at most one action per image under the exact Original-byte cap.
2. `SelectionContext-JointAction`:
   - first schedule `Static-JointAction` on fit at each budget;
   - for a calibration target, its same 32 fit context images now use the
     actions selected by that frozen fit schedule, with Edge as the default;
   - recompute action rewards and schedule calibration under the identical byte
     capacity.

This is deliberately a one-step selection-context diagnostic.  It asks whether
the selected-set state changes useful action marginals before investing in a
recurrent/iterative learned scheduler.

## Evaluation

- Primary mode: fixed-byte only.
- Budgets: 10%, 25%, and 40% of all-Original bytes for the same role.
- Report Official AP50-95, AP50, AP75, request KB/image, upload ratio, action
  mixture, Teacher objective, selection overlap, and source hashes.
- Teacher labels may use calibration GT because this is an explicit oracle
  ceiling diagnostic.  Calibration results cannot train or tune a deployable
  model.

## Success gate

`SelectionContext-JointAction` passes only if:

- it improves calibration Official AP50-95 over `Static-JointAction` by at
  least 0.002 at two of three budgets;
- every remaining budget is no worse than -0.001 AP;
- both methods respect the identical registered byte capacity at every budget;
- all isolation and source-hash checks pass.

Passing opens implementation of Selection-Contextual SharedAction.  Failure
stops this branch and retains SharedAction, CostEligible-AllROI, and
GBDT-Hierarchical as the action frameworks.

## Commands

```bash
python -m unittest discover \
  -s paper/experiments/20260805_coco_selection_context_jointaction_teacher_v1/tests -v
python paper/experiments/20260805_coco_selection_context_jointaction_teacher_v1/scripts/run_diagnostic.py --stage all
```

## Project structure

- `configs/protocol.json`: frozen roles, actions, reward, runtime, and gate.
- `scripts/teacher_core.py`: pure scheduling, context, and gate contracts.
- `scripts/run_diagnostic.py`: preflight, cached statistics/rewards, evaluation,
  and reporting.
- `tests/`: deterministic unit and protocol-contract tests.
- `runs/`: generated artifacts only.

## Boundaries

- Always: use existing caches; verify capacity and source hashes; checkpoint
  expensive stages; keep calibration separate from fit context selection.
- Ask first: change actions, budgets, context size, metric, or gate.
- Never: access val2017/final, retrain detectors/rankers, modify source
  experiments, or claim a deployable result from this oracle diagnostic.
