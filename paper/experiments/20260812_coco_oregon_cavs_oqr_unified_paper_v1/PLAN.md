# Implementation plan

## Phase 1 — frozen protocol and source audit

- [x] Validate dataset populations, identity disjointness, detector hashes,
  current@640 availability, Reversible manifests, five folds, and three seeds.
- [x] Fingerprint every prediction, score, checkpoint, and DCSB source read by
  the experiment.

## Phase 2 — same-checkpoint CAVS ablation

- [x] Assemble three-seed full-fit and five-fold OOF score ensembles.
- [x] Evaluate hard action masks O, OQ, and OQR with one capacity scheduler.
- [x] Prove action subsets, capacity, and score/image alignment in tests.

## Phase 3 — shared scalar action protocol

- [x] Train one fit-only SharedAction-OQR model per dataset.
- [x] Freeze its two thresholds once on calibration.
- [x] Load/retrain detector-specific Confidence, DCSB, ORIC, and MORIC scores.
- [x] Evaluate all scalar methods with the identical frozen action maps.

## Phase 4 — statistics and reporting

- [ ] Run the complete 1,000-replicate paired bootstrap matrix.
- [x] Generate unified point tables, action/cost breakdowns, source provenance,
  and a Chinese report.
- [ ] Verify the final completion manifest and all artifact hashes.

## Checkpoints

- Contract checkpoint: tests fail before implementation and pass afterward.
- Preflight checkpoint: no training/evaluation starts if any source drifts.
- Point checkpoint: all rows unique and byte-capacity safe.
- Final checkpoint: all bootstrap jobs complete and final was never selected on.
