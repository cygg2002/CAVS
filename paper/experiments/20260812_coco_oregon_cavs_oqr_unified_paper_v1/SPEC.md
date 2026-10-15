# Unified CAVS-OQR paper benchmark v1

## Purpose

This isolated experiment produces one internally consistent COCO/Oregon table
for the paper.  It answers three questions without changing any completed
source experiment:

1. Does one CAVS checkpoint benefit from expanding the action space from O to
   OQ and then to OQR?
2. Does the same `current@640` ROI action with Reversible Fusion transfer from
   COCO to Oregon Cloud-V3?
3. Does native joint image-action scheduling outperform scalar rankers when
   every scalar ranker receives one identical frozen SharedAction-OQR mapper?

## Frozen datasets and detectors

| Dataset | fit | calibration | evaluation | Cloud detector |
|---|---:|---:|---:|---|
| COCO | 9,678 | 571 | val2017 5,000 | canonical YOLO26x, SHA-256 pinned |
| Oregon | 9,923 | 4,961 | registered final 4,962 | Cloud-V3 epoch-28, SHA-256 pinned |

Both evaluation roles have already been used by earlier post-hoc experiments.
This run is paper-grade in reproducibility and statistical reporting, but is
not a new hidden confirmatory evaluation.

## Frozen method definition

The primary method is CAVS: an action-conditioned contextual value model plus
a byte-capacity scheduler.  One three-seed OQR checkpoint ensemble is evaluated
under three hard action masks:

- `CAVS-O`: Edge / Original;
- `CAVS-OQ`: Edge / Original / Q90;
- `CAVS-OQR`: Edge / Original / Q90 / `current@640`.

The O, OQ, and OQR rows must therefore differ only in allowed actions.  COCO
and Oregon both use `current@640` and Reversible Fusion with the same fusion
parameters (`match_iou=0.3`, `score_margin=0`, `add_threshold=0.1`).  Model
weights are dataset-specific and may never be transferred between detectors.

## Scalar baseline protocol

The paper table reports:

- Random (50 frozen seeds);
- Confidence;
- DCSB-Score-Adapted;
- ORIC-AP50 and MORIC-AP50 sensitivity reproductions;
- ORIC-AP5095 and MORIC-AP5095 primary external baselines.

Every scalar ranker uses the same dataset-specific SharedAction-OQR mapper.
The mapper is trained once on fit and its Q90/ROI thresholds are frozen once on
calibration.  No threshold may depend on ranker identity or budget.  Scalar
score scales are converted to calibration-reference empirical percentiles;
the same byte-capacity solver is used for all scalar methods and CAVS.
No final result may select a threshold, external variant, or method.

## Evaluation

- Fixed-byte 10%, 25%, and 40% is primary.
- Fixed-call 10%, 25%, and 40% is secondary.
- Report AP50-95, AP50, AP75, KB/image, capacity utilization, selected images,
  and Original/Q90/ROI action counts.
- Report Edge-only, Cloud-only Original, and Cloud-only Q90 as frozen detector
  references; these are not budget-matched scheduler competitors.
- Report fit 5-fold OOF for CAVS, calibration, and evaluation roles.
- Run 1,000 paired bootstrap replicates. COCO uses image resampling. Oregon
  uses event-block resampling as primary and image resampling as sensitivity.
- Bootstrap CAVS-OQR against CAVS-OQ, Confidence, DCSB, Random, ORIC-AP5095,
  MORIC-AP5095, and the calibration-frozen strongest ORIC/MORIC. Also compare
  CAVS-OQ against that same strongest external comparator.

## Source reuse boundary

Detector predictions and fully retrained dataset-specific checkpoints may be
reused only when role populations, action definitions, detector SHA-256,
fusion manifest, score/image alignment, seeds, folds, and artifact hashes all
match this specification.  Historical COCO MAUR selections and Oregon
Cloud-V1/Cloud-V2 learners are forbidden.  Source directories are read-only;
all new outputs live under this experiment.

## Acceptance criteria

- all protocol and unit tests pass;
- preflight records every required source SHA-256 and detects identity leakage;
- O/OQ/OQR are evaluated from the same checkpoint hashes;
- scalar action maps are bit-identical across rankers;
- no fixed-byte row exceeds its capacity and utilization is reported;
- the expected point and bootstrap matrices are complete and unique;
- final is absent from every training/threshold/variant selection input;
- a Chinese results report and machine-readable CSV/JSON artifacts are emitted.
- the report contains Fixed-byte, Fixed-call, O/OQ/OQR attribution, action-mix,
  reference bounds, and all paired-bootstrap confidence intervals.
