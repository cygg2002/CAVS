# Spec: COCO Frontier-Occupancy Gate Exploration v2

## Objective

Test whether a deployable learner can recover the GT-aware gain obtained by
removing ROI actions that never occupy the image's true
`contextual_reward - shadow_price * bytes` upper envelope.

This is a post-hoc fit-OOF exploration triggered by v1's error audit. It is not
a confirmatory or paper-final result.

## Frozen evidence

- Source candidate: v1 three-seed `OQR/BoundaryPathDistill` ensemble.
- Canonical COCO train2017 fit=9,678 only.
- GT any-frontier ROI gate ceiling over v1: +0.002766/+0.003230/+0.002831
  AP50-95 at Fixed-byte 10%/25%/40%.
- No calibration, val2017, audit, final, or registered-final access.

## Methods

- `Ungated-AFPD`: frozen v1 ensemble scores.
- `CostFree-FrontierGate`: predicts whether ROI ever belongs to the true
  parametric action frontier, without payload/resource inputs.
- `CostAware-FrontierGate`: predicts the same target while allowing the gate
  (not the Value model) to observe known action payload/resource features.
- `GT-AnyFrontier-Gate`: non-deployable upper-bound reference only.

Both learned gates use a frozen probability threshold of 0.5. No threshold is
selected on OOF results.

## Novelty boundary

The contribution under test is not ordinary ROI safety classification. The
target is budget-independent *membership in the complete parametric
shadow-price frontier*. Value remains cost-free; known cost is optionally
exposed only to a constraint/eligibility model. A paper claim requires broader
literature review and independent data confirmation.

## Protocol

- Same five outer folds as v1; one frozen seed for the first screen.
- OOF targets are generated only from each target's fold-contained reward.
- Fixed-byte 10%/25%/40% primary; Fixed-call secondary.
- Train only on fit folds. No detector inference and no GPU.
- Gate success versus `Ungated-AFPD`: at least two Fixed-byte gains >=0.001,
  remaining point >=-0.001, all Fixed-call points >=-0.001, capacity >=98%.

## Success boundary

A learned gate is interesting only if it improves actual AP, not merely ROI
classification. If neither method passes, retain ungated AFPD and record the
occupancy head as a negative result.

