# Spec: AFPD Full Retrain and Full-Val Paper-Format Evaluation

## Objective

Independently retrain AFPD and Occupancy from scratch under the canonical COCO
protocol, then measure them on all 5,000 val2017 images.  This run answers
whether the 571-image calibration failure was representative.  It is a full
paper-format reproduction, but not a new hidden final because val2017 has
already been reused by historical experiments.

## Fixed inputs

- Source protocol: `20260810_coco_afpd_occupancy_paper_protocol_v1`.
- Training population: canonical train2017 fit=9,678.
- Training: 5-fold OOF x 3 seeds, followed by full fit=9,678 training x 3 seeds.
- Models: four action methods and two Occupancy gates, all trained from scratch.
- Evaluation population: canonical val2017=5,000.
- Actions, payloads, detector predictions, metrics, budgets, and scheduler are
  inherited unchanged from the source protocol.
- Frozen external selections are inherited from the completed full scheduler
  benchmark.

## Reported methods

- Current-OQ
- Current-OQR
- AFPD-OQ
- AFPD-OQR
- AFPD-OQR+OccupancyFree
- AFPD-OQR+OccupancyAware

Raw ActionValue-OQ and MORIC/ORIC remain frozen external comparisons; they are
not retrained by this experiment.

## Training and statistical protocol

- Fixed-byte 10%/25%/40% is primary.
- Fixed-call 10%/25%/40% is secondary.
- Run 1,000 paired image-level bootstrap replicates for AFPD-OQR versus Raw
  ActionValue-OQ, MORIC, and ORIC, plus both Occupancy gates versus AFPD-OQR.
- Never tune or select a method using these results.
- Calibration is evaluated once and its failed/pass status is reported, but the
  predeclared full-val evaluation is unconditional in this post-hoc protocol.

## Boundaries

- Always: start with an empty checkpoint directory; hash source code and reward
  caches; use the complete 5,000-image manifest; preserve exact byte capacity;
  disclose the calibration gate result.
- Never: modify the source experiment; import its trained checkpoints; tune on
  calibration/val2017; call this a hidden final; use val2017 to choose a new
  AFPD variant.

## Success criteria

1. Exactly 60 OOF action jobs, 30 OOF Occupancy jobs, 12 full-fit action jobs,
   and 6 full-fit Occupancy jobs train from scratch.
2. Exactly 36 aggregate calibration rows and 36 full-val rows are produced.
3. Every Fixed-byte row respects capacity.
4. Exactly 15 bootstrap comparisons are produced (5 pairs x 3 budgets), each
   with 1,000 replicates.
5. The report states `confirmatory=false`, `calibration_gate_failed=true`, and
   `final_is_hidden=false`.
