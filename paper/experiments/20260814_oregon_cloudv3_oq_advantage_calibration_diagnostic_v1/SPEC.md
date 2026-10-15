# Spec: Oregon Cloud-V3 O/Q Advantage Calibration Diagnostic v1

## Objective

Determine whether the Oregon Cloud-V3 high-budget regression can be repaired
without changing or retraining the CAVS network.  The experiment calibrates the
Original-versus-Q90 advantage from frozen three-seed, five-fold OOF scores and
then evaluates frozen calibration and already-consumed final roles.

This is a post-hoc diagnostic.  Final results may explain the mechanism and
select the next registered method, but are not confirmatory paper evidence.

## Frozen matrix

- Source learner/rewards: `20260813_coco_oregon_cavs_final_completion_v1`.
- Raw point reference: `20260814_oregon_cloudv3_current640_maur_point_only_v1`.
- Roles evaluated: calibration and final.
- Training source for every calibrator: fit OOF only.
- Budgets: Fixed-byte 10%, 25%, and 40%.
- Fusion: MAUR.
- Methods:
  1. `Raw-CAVS-OQR`.
  2. `PerAction-Isotonic-OQR`.
  3. `Advantage-Isotonic-Midpoint-OQR`.
  4. `Advantage-Isotonic-QAnchor-OQR`.
  5. `Calibration-Frozen-OQMargin-OQR`.
  6. `Q90-Preferred-Diagnostic`.

The Q90-preferred row is a diagnostic boundary, not the proposed deployable
method.  It suppresses Original whenever Q90 is available and leaves ROI
unchanged.

The O/Q margin is a single global value shared by all three budgets. Candidate
margins are frozen from fit-OOF advantage quantiles. Calibration Official AP is
used once to choose among them subject to low-budget non-inferiority and
40%-versus-25% monotonicity; final is never used for this choice.

## Commands

```bash
python -m unittest discover -s tests -v
nice -n 19 python scripts/run_diagnostic.py
```

## Project structure

- `configs/protocol.json`: frozen source, roles, methods, and gates.
- `scripts/calibration_core.py`: pure O/Q score transformations.
- `scripts/run_diagnostic.py`: read-only source loading, OOF calibration, scheduling, and evaluation.
- `tests/`: protocol and score-transformation tests.
- `runs/`: generated outputs only.

## Testing strategy

- Prove Q-anchor calibration preserves Q90 scores exactly.
- Prove midpoint calibration preserves the O/Q midpoint exactly.
- Prove both methods realize the requested calibrated advantage.
- Prove Q90-preferred suppression never changes Edge, Q90, or ROI scores.
- Prove a frozen margin changes only the Original score by the requested amount.
- Reject any protocol that fits on calibration/final or modifies the source.

## Boundaries

### Always

- Reuse frozen Cloud-V3 predictions, rewards, OOF scores, and full-fit scores.
- Fit calibration only from fit OOF scores and fit rewards.
- Preserve the source experiment byte-for-byte.
- Mark final as already consumed and every conclusion as exploratory.

### Never

- Retrain the detector or learner.
- Read Cloud-V4 scores or checkpoints.
- Select a method or threshold from final.
- Claim that a final improvement is confirmatory.

## Success criteria

The calibration-only repair is promising if one deployable calibrated method:

1. has final Fixed-byte 40% AP at least as high as its own 25% AP;
2. is no worse than Raw by more than 0.001 AP at 10% or 25%;
3. improves Raw at 40% by at least 0.001 AP;
4. uses at least 98% of every byte capacity;
5. is no worse than Raw by more than 0.001 AP on every calibration budget and
   has calibration 40% AP at least as high as calibration 25% AP.

If no method passes, post-hoc calibration is insufficient and the next method
must change the training objective or observable input information.
