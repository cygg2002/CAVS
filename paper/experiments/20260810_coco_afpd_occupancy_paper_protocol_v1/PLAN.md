# Plan: COCO AFPD + Occupancy Paper Protocol v1

## Phase 1: Freeze contracts

- [x] Define the 2x2 objective/action attribution matrix.
- [x] Freeze AFPD v1 as primary and Occupancy v2 as secondary.
- [x] Mark val2017 as post-hoc reuse.
- [ ] Add failing unit tests for protocol, gates, attribution, and row counts.

## Checkpoint A

- [ ] Tests fail because the new core implementation is absent.

## Phase 2: 2x2 OOF attribution

- [ ] Reuse canonical folds and v1 training functions read-only.
- [ ] Train Current/AFPD in OQ/OQR for three seeds x five folds.
- [ ] Evaluate Fixed-byte and Fixed-call at 10%/25%/40%.
- [ ] Train and evaluate three-seed CostFree/CostAware Occupancy gates.

## Checkpoint B

- [ ] Exactly 96 action-model rows and 72 occupancy-policy rows.
- [ ] Source seed reproduction agrees with frozen v1 within tolerance.
- [ ] No calibration/final access in OOF artifacts.

## Phase 3: Full-fit and one-time calibration

- [ ] Train 12 action checkpoints and six occupancy checkpoints on fit only.
- [ ] Predict calibration once; evaluate six frozen ensemble methods.
- [ ] Import frozen external calibration references without refitting them.
- [ ] Apply the predeclared primary calibration gate.

## Checkpoint C

- [ ] Exactly 36 calibration rows, complete action distributions, and capacity
  utilization >=98% for Fixed-byte.

## Phase 4: Conditional post-hoc final and bootstrap

- [ ] If the primary calibration gate passes, predict/evaluate val2017 once.
- [ ] Compare AFPD-OQR to Raw ActionValue-OQ and MORIC-MLP with paired
  image-level bootstrap (1,000 replicates).
- [ ] Compare Occupancy policies to AFPD-OQR as secondary ablations.
- [ ] Write Chinese results with explicit evidence tiers and caveats.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| OQ/OQR gain is conflated | Mandatory 2x2 matrix and interaction report |
| v2 cherry-picks a gate | Both frozen variants reported; primary cannot change |
| final reuse is overstated | Hard-code post-hoc status in protocol and report |
| image/action order mismatch | Exact ID/order assertions for every score artifact |
| CPU oversubscription | One torch thread per process and bounded workers |
| source experiments are mutated | New isolated output tree; sources read-only |

