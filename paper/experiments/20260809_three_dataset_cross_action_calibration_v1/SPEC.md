# Spec: Three-Dataset Cross-Action Calibration Diagnostic v1

## Objective

Determine whether the remaining failure of `ActionValue-OQ-Joint` is caused by
miscalibration between the Original and Q90 action values, or by image
membership ranking.  The experiment is cache-only and covers COCO, ENA24, and
Oregon under the already frozen canonical fit/calibration roles.

Only `Raw-ActionValue-OQ` and `PerAction-Isotonic-OQ` are deployable.  Oracle
rows are diagnostic ceilings and must never be presented as deployable gains.

## Frozen experiment matrix

- Datasets: `coco`, `ena24`, `oregon`.
- Roles: `fit_oof`, `calibration`.
- Modes: `fixed_byte` (primary), `fixed_call` (secondary).
- Budgets: `0.10`, `0.25`, `0.40`.
- Actions: `edge`, `original`, `q90`; ROI is forbidden.
- Methods:
  1. `Raw-ActionValue-OQ`.
  2. `PerAction-Isotonic-OQ`.
  3. `CalibratedMembership-OracleAction`.
  4. `OracleMembership-CalibratedAction`.
  5. `FullOracle-OQ`.
- Expected rows: 60 per dataset, 180 total.

## Calibration semantics

- Fit one monotone calibrator for each `(budget, action)` pair.
- Original and Q90 are calibrated independently; Edge remains zero.
- Use 64 deterministic quantile bins and PAVA.
- `fit_oof` predictions are produced with a second cross-fitting layer: each
  calibrator is fit on four folds and applied to the held-out fold.
- `calibration` predictions use calibrators fit on all fit-OOF scores/rewards.
- No hyperparameter or method selection may use calibration.

## Oracle decomposition

- `CalibratedMembership-OracleAction`: preserve the image IDs selected by the
  calibrated method and optimize Original/Q90 using contextual ground-truth
  rewards within that fixed membership.
- `OracleMembership-CalibratedAction`: preserve the image IDs selected by the
  full OQ teacher and choose Original/Q90 using calibrated values.
- `FullOracle-OQ`: jointly select membership and action using contextual
  ground-truth rewards.
- Fixed-byte Oracle rows use the same byte capacity as deployable rows.  A
  fixed-membership Oracle must assign one cloud action to every preserved image
  and may not silently drop images.

## Sources

- Canonical roles, OQ scores, contextual rewards, schedulers, and evaluators
  are read from completed frozen experiments.
- The primary reusable source is
  `20260807_three_dataset_actionvalue_roi_gate_v1`.
- No detector inference, GPU, new reward construction, or final role is used.
- Oregon remains on `canonical_oregon_event_v1`; Cloud-V2 sources are forbidden.

## Commands

```bash
python -m unittest discover -s tests -v
python scripts/run_diagnostic.py --dataset coco --stage preflight
python scripts/run_diagnostic.py --dataset ena24 --stage preflight
python scripts/run_diagnostic.py --dataset oregon --stage preflight
python scripts/run_parallel.py
```

## Project structure

- `configs/protocol.json`: frozen matrix, calibration settings, and gate.
- `scripts/calibration_core.py`: pure calibration and result contracts.
- `scripts/run_diagnostic.py`: source validation, calibration, scheduling, and
  evaluation for one dataset.
- `scripts/run_parallel.py`: resumable three-dataset CPU launcher.
- `tests/`: pure unit and protocol tests.
- `runs/`: generated artifacts only.

## Testing strategy

- Unit-test PAVA monotonicity and deterministic interpolation.
- Prove cross-fitting never trains on the held-out indices.
- Prove fixed-membership action selection retains exactly the requested IDs.
- Prove ROI/final sources and incomplete result matrices are rejected.
- Prove fixed-byte capacity and fixed-call cardinality invariants.
- Run preflight for all datasets before launching evaluation.

## Boundaries

### Always

- Align scores and rewards in each role's canonical sample order.
- Use role-local identities for split isolation.
- Record source hashes and generated artifact hashes.
- Mark all Oracle rows `deployable=false` and `gt_aware=true`.

### Ask first

- Changing actions, roles, budgets, bin count, folds, or gate thresholds.
- Reading a final role or adding a new dependency.

### Never

- Modify a completed source experiment.
- Fit anything on calibration or final.
- Read Oregon Cloud-V2/Profile-Scale caches.
- Claim Oracle rows as a deployable result.

## Success criteria

The deployable calibrated method passes only if all conditions hold on frozen
calibration:

1. Every fixed-byte delta versus Raw is at least `-0.001` AP.
2. ENA24 40% fixed-byte AP is at least `0.7095`.
3. Oregon fixed-byte is no worse than Raw by more than `0.001` at any budget.
4. Fixed-call is no worse than Raw by more than `0.001` at any dataset/budget.
5. At least two datasets have at least two fixed-byte gains of `+0.001` AP.
6. All capacity, cardinality, split-isolation, and no-final checks pass.

COCO comparison against external ORIC/MORIC is intentionally deferred until a
deployable candidate passes this development gate; this diagnostic does not
reuse val2017/final to make that choice.

## Open questions

None.  Failure closes the static single-frame calibration branch; success
freezes the calibrated scheduler for a separately registered paper evaluation.
