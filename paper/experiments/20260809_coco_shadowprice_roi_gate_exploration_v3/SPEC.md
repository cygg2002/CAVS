# Spec: COCO Shadow-Price-Conditioned ROI Gate v3

## Question

Can a gate decide whether ROI is locally optimal at the *current global byte
shadow price*, instead of learning v2's overly broad "ROI is useful somewhere
on the frontier" label?

## Frozen design

- Source Value scores: v1 three-seed OQR/BoundaryPathDistill ensemble.
- Canonical fit=9,678, same five OOF folds, one screen seed.
- Cost-aware Edge/action features; the gate additionally receives normalized
  shadow price through the existing budget input.
- Training price grid fixed before results:
  `[0, .02, .05, .10, .20, .40, .80, 1.60]`.
- Inference price is the ungated AFPD scheduler's own deployable lambda times
  the dataset-wide mean Original bytes. One-pass gating, then one reschedule.
- Probability threshold fixed at 0.5; no OOF threshold tuning.

## Comparators

- Ungated AFPD.
- v2 CostAware Any-Frontier Gate.
- Price-Conditional Gate (v3).
- GT gate at the learned scheduler price (non-deployable ceiling).

## Boundary

Fit-OOF post-hoc exploration only. No calibration, val2017, audit, final,
registered-final, detector inference, or GPU. Success uses the same AP/capacity
gate as v1/v2.

