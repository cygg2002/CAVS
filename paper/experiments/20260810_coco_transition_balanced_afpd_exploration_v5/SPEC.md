# Transition-Balanced AFPD: frozen diagnostic specification

## Question

Does the existing Action-Frontier Path Distillation (AFPD) objective give
ROI-related frontier transitions materially less training mass than the other
actions, despite ROI having a useful Teacher frontier on many ROI-available
images?

This is a fit-only diagnostic. It reuses the frozen COCO canonical fit cache
and the exact OQR frontier probes from AFPD v1. It does not train a model,
rerun a detector, access calibration, or access val2017/final.

## Frozen measurements

- ROI availability over all fit images.
- Valid before/after crossing probes and their existing boundary weights.
- Fraction of valid crossing probes whose generating pair contains ROI.
- Target-action frequency and boundary-weight mass, both globally and among
  ROI-available images.
- ROI occurrence on the exact Teacher upper envelope.

Action columns are resolved from each record's `action_id`; ROI is never
assumed to occupy a fixed tensor column.

## Go/no-go rule fixed before reading the result

Proceed to one fit-only, one-seed Transition-Balanced AFPD screen only when:

1. ROI is on the Teacher frontier for at least 40% of ROI-available images;
2. ROI receives less than 15% of the existing global target boundary-weight
   mass; and
3. among ROI-available images, ROI-involved crossings receive less than 35%
   of valid crossing weight (the unfiltered four-action pair share is 50%).

Otherwise stop this branch. These thresholds diagnose a large imbalance;
they are not tuned for downstream AP.

## Integrity constraints

- canonical fit population: 9,678 images;
- action space: Edge/Original/Q90/ROI;
- all source paths and protocol values inherited from AFPD v1;
- calibration, audit, val2017, and registered final remain inaccessible;
- output is descriptive evidence, not a paper-level performance claim.

