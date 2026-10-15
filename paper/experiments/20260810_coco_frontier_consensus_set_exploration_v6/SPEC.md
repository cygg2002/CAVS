# Frontier Consensus Set (FCS): fit-only exploration

## Motivation

AFPD v1 has three independently trained OOF seeds. Averaging their values
hides whether an action is consistently believed to lie on the image's
reward-byte upper envelope. This experiment asks whether a set-valued action
prediction can remove unreliable actions before exact capacity scheduling.

## Frozen variants

- `AFPD-Mean`: the existing three-seed mean; unchanged baseline.
- `ROI-FCS-2of3`: retain ROI only when at least two seeds place it on their
  predicted parametric frontier. Edge, Original, and Q90 are unchanged.
- `AltAction-FCS-2of3`: retain Q90 and ROI only when at least two seeds place
  the action on their predicted frontier. Edge and universally available
  Original are retained as the no-upload and cloud-call fallbacks.

Each seed's frontier is computed from all valid pairwise crossings plus price
endpoints, using the same normalized payload scale and price range as AFPD v1.
No threshold or candidate is selected using downstream AP.

## Protocol

- COCO canonical fit=9,678 only, strict five-fold OOF scores;
- OQR actions: Edge/Original/Q90/ROI, with missing actions respected;
- Fixed-byte 10%/25%/40% primary, Fixed-call secondary;
- exact existing scheduler, cached detector predictions, no detector inference;
- no calibration, audit, val2017, or final access.

## Frozen screen gate

Relative to `AFPD-Mean`, a candidate passes only if at least two Fixed-byte
points improve by >=0.001 AP, every remaining Fixed-byte point is >=-0.001,
and all Fixed-call points are >=-0.001.

## Novelty boundary

Ensembles, majority consensus, uncertainty sets, and set-valued prediction are
established ideas. The experiment only tests the specific combination of
per-image parametric frontier membership and byte-constrained multi-action
offloading. A positive result would still require a broader literature search
and independent evaluation before any novelty claim.
