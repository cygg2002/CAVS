# Spec: COCO ActionValue ROI Contribution Diagnostic

## Objective

Determine whether ROI contributes real system-level AP beyond the learned
Original/Q90 scheduler, and whether a perfect relative ROI eligibility gate
would expose additional recoverable value.

## Frozen Inputs

- Source scores: canonical fit=9,678 five-fold OOF and calibration=571 from
  `20260805_coco_actionconditioned_valuerisk_oof_v1`.
- Actions: Edge, Original, Q90, and the existing ROI cache only.
- Budgets: fixed-byte 10%, 25%, and 40%; fixed-call is checked for selection
  invariance because the source scheduler selected no ROI there.
- Metrics: Official AP50-95, AP50, AP75, KB/image, action counts, harmful and
  relatively-suboptimal ROI rates.

## Compared Policies

1. `ActionValue-OQ`: disable ROI and reschedule Original/Q90.
2. `ActionValue-OQR`: unchanged learned ActionValue with all three cloud actions.
3. `ActionValue-OQR-OracleRelativeGate`: expose ROI only when its cached
   contextual reward is positive and strictly greater than every available
   Original/Q90 alternative for that image. This is a GT-aware diagnostic
   ceiling, not deployable.

## Success Criteria

- Every role has 3 policies x 3 fixed-byte budgets with valid capacities.
- OOF covers 9,678 unique images; calibration covers 571 images.
- Report full-minus-OQ and Oracle-minus-OQ AP deltas.
- Retain ROI research only if Oracle-minus-OQ is at least +0.001 AP at two
  budgets and no worse than -0.001 at the remaining budget in both roles.
- Treat learned ROI as already effective only under the same rule for
  full-minus-OQ, with no material regression in ROI safety.
- Never access audit, val2017 final, or registered final.

## Boundaries

- Always: bind outputs to source hashes and report GT-aware status explicitly.
- Never: retrain, tune thresholds on calibration, add actions, or access final.
