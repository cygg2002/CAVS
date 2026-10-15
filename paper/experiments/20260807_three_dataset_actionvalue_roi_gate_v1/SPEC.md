# Spec: Three-dataset ActionValue ROI advantage diagnostic

## Objective

Determine whether canonical ROI adds deployable value to the existing
ActionValue Original/Q90 joint scheduler on COCO, ENA24, and Oregon.

Compare the same three methods on fit OOF and frozen calibration:

1. `ActionValue-OQ-Joint`: Edge, Original, and Q90 only.
2. `ActionValue-OQR-Naive`: ROI is an equal fourth action.
3. `ActionValue-OQR-AdvantageGate`: ROI is available only when its predicted
   value exceeds the best predicted Original/Q90 value by an OOF-frozen margin.

## Assumptions

- COCO uses canonical fit=9,678 and calibration=571.
- ENA24 uses canonical fit=1,400 and calibration=400.
- Oregon uses `canonical_oregon_event_v1`, fit=9,923 and calibration=4,961.
- COCO OQ/OQR scores are reused from the completed canonical ActionValue run.
- ENA24/Oregon OQ scores are reused from the completed cross-dataset run;
  only the OQR model is newly trained.
- No detector inference and no final role access are allowed.

## Commands

```bash
python -m unittest discover -s tests -v
python scripts/run_roi_gate.py --dataset coco --stage all --workers 8
python scripts/run_roi_gate.py --dataset ena24 --stage all --workers 24
python scripts/run_roi_gate.py --dataset oregon --stage all --workers 24
```

## Project structure

```text
configs/protocol.json       Frozen dataset, model, and gate protocol
scripts/protocol_core.py    Pure contracts and gate logic
scripts/run_roi_gate.py     Resumable cache/reward/train/evaluation pipeline
tests/                      Unit and protocol tests
runs/<dataset>/             Independent checkpoints, scores, and results
notes/                      Generated result summary
```

## Method and gate

The OQR learner uses the same cost-free ActionValue architecture, detection-set
representation, contextual AP50-95 reward, folds, and seeds as the OQ baseline.
The only new input action is canonical ROI.

For image `i`, predicted ROI advantage is:

```text
adv_i = V(i, ROI) - max(V(i, Original), V(i, Q90))
```

For each mode/budget work point, OOF predictions select the threshold that
maximizes summed contextual reward while keeping selected harmful ROI at or
below 10%. This threshold is then frozen and applied once to calibration.
If no finite threshold satisfies the safety constraint, ROI is disabled.

## Metrics

- Primary: Fixed-byte AP50-95 at 10%, 25%, and 40%.
- Secondary: Fixed-call AP50-95 at the same budgets.
- AP50, AP75, KB/image, selected image count, capacity utilization.
- Original/Q90/ROI counts and ROI ratio.
- Harmful ROI rate and relative-suboptimal ROI rate.
- Selected contextual reward and Teacher regret.
- AP deltas against OQ.

## Testing strategy

- Unit tests for protocol constants, method matrix, ROI score masking, threshold
  selection, and result matrix completeness.
- Remote preflight verifies role counts, disjoint identities, cache files, and
  source hashes without reading final.
- A small real-cache smoke run precedes full background launch.
- All stages use atomic outputs and completion markers.

## Boundaries

### Always

- Use fit OOF only to select AdvantageGate thresholds.
- Apply frozen thresholds once to calibration.
- Preserve the exact canonical action geometry and evaluator.
- Keep Fixed-byte selection within capacity.

### Never

- Modify completed source experiments.
- Read final, audit, or registered final.
- Use Oregon Cloud-V2 in this canonical diagnostic.
- Retrain detectors or change ROI geometry, codec, fusion, reward, or split.
- Describe the OOF-selected threshold result as an independent final result.

## Success criteria

- All tests, preflights, and real-cache smoke tests pass.
- Each dataset produces 36 unique rows:
  2 roles x 3 methods x 2 modes x 3 budgets.
- Every calibration AdvantageGate row records an OOF-frozen threshold.
- All Fixed-byte capacities are safe and all Fixed-call counts are exact.
- No source provenance contains a final or Cloud-V2 path.

## Interpretation gate

ROI strengthens the unified scheduler only if frozen calibration shows, for a
dataset, at least two Fixed-byte improvements of at least 0.001 AP over OQ,
the remaining point is no worse than -0.001 AP, and harmful ROI does not exceed
10%. COCO is allowed to converge to zero ROI; ENA24 is expected to use ROI more
actively; canonical Oregon should remain conservative.
