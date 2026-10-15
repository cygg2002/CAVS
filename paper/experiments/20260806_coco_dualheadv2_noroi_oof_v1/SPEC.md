# Spec: COCO DualHead-v2 No-ROI OOF Diagnostic

## Objective

Run one cache-only COCO development diagnostic with two gated questions:

1. Does the already-computed selection-context teacher improve when ROI is
   removed and only Original/Q90 actions remain?
2. If not, can the existing action-value mean/lower-quantile heads be deployed
   as a non-collapsing DualHead-v2 selector that improves ActionValue-OQ?

This experiment does not train detectors, regenerate rewards, or access
COCO val2017/final. It reuses the canonical train2017 `fit=9,678` five-fold OOF
scores and evaluates frozen choices once on `calibration=571`.

## Data and actions

- Fit OOF: canonical COCO train2017, exactly 9,678 images.
- Frozen calibration: canonical COCO train2017, exactly 571 disjoint images.
- Candidate actions: Original and Q90 only; Edge is the zero-cost fallback.
- ROI is hard-masked in every reward schedule, learned schedule, result row,
  and saved selection.
- Primary mode: Fixed-byte at 10%, 25%, and 40% of all-Original bytes.
- Secondary mode: Fixed-call at the same nominal budgets.

## Methods

- `ActionValue-OQ`: frozen action-conditioned value scores.
- `RiskMean-OQ`: mean head from the frozen value-risk model.
- `DualHeadV2-OQ`: for each cloud action,

  `score = mean - alpha * max(mean - lower_quantile, 0)`.

The alpha grid is evaluated only on fit OOF. For each budget, the best OOF
Fixed-byte alpha is frozen subject to at least 98% byte-capacity utilization
and Fixed-call non-inferiority to `ActionValue-OQ`. Calibration cannot select
or modify alpha.

## Selection-context gate

Reuse the completed static and one-step selection-context reward caches from
`20260805_coco_selection_context_jointaction_teacher_v1`, but reschedule only
Original/Q90. The branch passes only if selection context improves calibration
AP50-95 by at least 0.002 at two budgets, every other point is no worse than
-0.001, and every byte capacity is respected.

Failure records the negative result and skips SC-Value/SC-VC training.

## DualHead-v2 gate

Relative to `ActionValue-OQ`, DualHead-v2 passes a role only if:

- at least two Fixed-byte budgets improve AP50-95 by at least 0.001;
- the remaining Fixed-byte point is no worse than -0.001;
- every Fixed-call point is no worse than -0.001;
- every Fixed-byte selection uses at least 98% of capacity without exceeding
  it; and
- no saved selection contains ROI.

A later three-seed protocol opens only if both fit OOF and frozen calibration
pass. This single-seed diagnostic is not a paper final.

## Commands

```bash
python -m unittest discover \
  -s paper/experiments/20260806_coco_dualheadv2_noroi_oof_v1/tests -v
python paper/experiments/20260806_coco_dualheadv2_noroi_oof_v1/scripts/run_diagnostic.py --stage all
```

## Project structure

- `configs/protocol.json`: frozen roles, actions, alpha grid, gates, and CPU use.
- `scripts/dualheadv2_core.py`: pure blending, alpha-freezing, No-ROI, and gate contracts.
- `scripts/run_diagnostic.py`: source validation, scheduling, parallel COCO evaluation, and reporting.
- `tests/`: deterministic unit and protocol tests.
- `runs/`: generated checkpoints, logs, selections, and reports.

## Testing strategy

- Unit-test score blending, alpha choice, gate logic, No-ROI enforcement, and
  protocol isolation.
- Integration preflight verifies frozen source completions, score/image order,
  source hashes, role counts, and forbidden-role isolation.
- Result checks require a complete, duplicate-free matrix and exact capacity
  accounting.

## Boundaries

- Always: use existing read-only caches; freeze alpha on fit OOF; hash sources;
  checkpoint every stage; report negative gates.
- Ask first: change actions, alpha grid, budgets, gates, roles, or source scores.
- Never: access audit/val2017/final/registered-final, train on calibration,
  modify source experiments, or describe this as a confirmatory result.

## Success criteria

- All tests pass and preflight confirms fit=9,678/calibration=571.
- No ROI action appears in any schedule or result.
- Selection-context No-ROI gate is recorded before the SC branch is skipped.
- OOF alone freezes exactly one alpha per budget.
- Calibration evaluates those frozen alphas once.
- Completion artifacts explicitly state that no final role was accessed.

