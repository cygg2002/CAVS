# COCO/Oregon Full Scheduler Benchmark v1

## Objective

Run one cache-only, post-hoc benchmark that places the strongest frozen internal
scheduler beside simple, historical, and external baselines on COCO and Oregon.
The experiment also indexes the already-completed ROI/fusion and Oregon Cloud-V2
diagnostics without mixing incompatible detector protocols.

## Status and leakage boundary

- Result status: `posthoc_final_reuse_full_benchmark`.
- Training uses canonical `fit` only.
- Isotonic action calibration uses fit OOF predictions and fit targets only.
- Fixed-byte lambda selection uses `calibration` only.
- Final is evaluation-only and may not select a model, threshold, comparator, or
  action policy.
- No detector inference and no GPU are required.

## Canonical datasets

| Dataset | Canonical run | fit | calibration | final |
|---|---|---:|---:|---:|
| COCO | `canonical_v1` | 9,678 | 571 | 5,000 val2017 |
| Oregon | `canonical_oregon_event_v1` | 9,923 | 4,961 | 4,962 |

The canonical Oregon main table uses Cloud-V1. Cloud-V2 sources are allowed only
in the isolated `cloudv2_reference` table and are never consumed by a canonical
evaluation function.

## Track A: primary frozen comparison

Modes are Fixed-byte (primary) and Fixed-call (secondary), each at 10%, 25%, and
40%. The primary action space is Edge/Original/Q90 (strictly no ROI).

Internal candidates:

- Raw ActionValue-OQ Joint
- Per-Action Isotonic ActionValue-OQ Joint
- DualHead + SharedAction-NoROI
- SetValue + SharedAction-NoROI
- Current-BudgetSet + SharedAction-NoROI

Historical/simple baselines:

- Random (50 seeds)
- Confidence
- ClassGainPrior
- M2E
- Confidence-Residual M2E
- SemanticRank
- MORIC-style
- GEC-style
- Current-Controlled-32D

External baselines:

- DCSB-Native and DCSB-Score-Adapted
- ORIC-MLP
- MORIC-MLP

Scalar Rankers are evaluated under both Original-only and
SharedAction-NoROI. ActionValue directly schedules Original/Q90 under the same
byte or call capacity. The latter is a system-level comparison, not a pure
ranker-only comparison.

## Track B: canonical action-policy extension

Import and fingerprint the completed final rows for SharedAction,
CostEligible-AllROI, and GBDT-Hierarchical. Include DualHead and all historical
Rankers where those rows already exist. Do not refit SafeROI GBDT thresholds.

## Track C: ROI and fusion diagnostics

Import completed ActionValue-OQR, MAUR, and Reversible results. Any role that is
fit OOF or calibration remains labelled diagnostic and must not be promoted to a
final result. Missing cache-compatible rows may be recomputed; no detector cache
may be generated.

## Track D: Oregon Cloud-V2 reference

Import the completed Profile x Scale teacher ceiling and MAUR/Reversible audit
results. These rows carry `detector_protocol=cloud_v2` and
`comparable_to_primary=false`.

## Architecture boundary

There is no frozen definition for `DualHead-v3`; this experiment must not invent
one after seeing final results. It imports the completed DualHead-v2,
RelActionRank, and Factorized Global ActionRank diagnostics as development
references instead.

## Statistics

- 1,000 paired bootstrap replicates for the primary candidate against the
  strongest frozen external method, Confidence, and Random at each primary work
  point.
- COCO resampling unit: image.
- Oregon resampling unit: event/camera group where recoverable; image bootstrap
  is reported only as a sensitivity analysis.
- Random uses 50 fixed seeds beginning at 20260720.

## Success gate

For Calibrated ActionValue-OQ versus the strongest ORIC/MORIC system at each
dataset's same Fixed-byte capacity:

- all six deltas must be at least -0.001 AP;
- at least four of six deltas must be at least +0.001 AP;
- each dataset must contribute at least one +0.001 point;
- capacity utilization must be in [0.98, 1.00].

This gate is descriptive because both finals have already been reused. It cannot
create a new confirmatory claim.

