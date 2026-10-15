# Spec: COCO Dual-Conditioned Frontier Residual v4

## Question

Can a one-pass residual model correct AFPD's ROI value using both the global
byte shadow price and the first-pass ROI-versus-best-alternative frontier
state, without disturbing the frozen Edge/Original/Q90 values?

## Method

`DualConditioned-ROIResidual` first schedules the frozen three-seed AFPD
ensemble.  For every ROI-available image it then observes only deployable
quantities:

- the first-pass normalized shadow price;
- AFPD's cost-adjusted ROI margin over the best Edge/Original/Q90 alternative;
- the ROI-minus-alternative normalized byte-cost gap;
- the identity of that best alternative;
- the existing Edge detection set, scene, and ROI action geometry.

The model predicts a residual to the frozen ROI value.  It is supervised by
the true contextual-reward utility-margin residual on fit OOF only.  At
deployment the residual changes only ROI's cost-free value, then the exact
capacity scheduler is run once more.  Edge, Original, and Q90 values remain
bit-for-bit frozen.

## Frozen screen

- Canonical COCO train2017 fit=9,678; exact existing five OOF folds.
- Source scores: v1 OQR/BoundaryPathDistill three-seed OOF ensemble.
- Training prices: `[0, .02, .05, .10, .20, .40, .80, 1.60]`.
- One screen seed, five folds, CPU only.
- Loss: Smooth-L1 residual plus fixed 0.25 sign loss at temperature 0.05.
- Fixed-byte 10%/25%/40% primary; Fixed-call safety.
- Compare frozen AFPD, v2 any-frontier gate, v3 price gate, and GT correction.

## Gate and boundary

The success gate is unchanged from v1-v3: at least two Fixed-byte gains of
0.001 AP, every other point no worse than -0.001 AP, Fixed-call no worse than
-0.001 AP, and at least 98% byte-capacity utilization.  Passing opens a
three-seed confirmation; failure ends this branch.  No calibration, val2017,
audit, final, registered final, detector inference, threshold tuning, or GPU.

