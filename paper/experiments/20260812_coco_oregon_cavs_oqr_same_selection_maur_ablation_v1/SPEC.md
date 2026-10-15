# Spec: CAVS-OQR same-selection MAUR fusion ablation

## Objective

Determine whether the historical MAUR fusion itself is more accurate than the
current Reversible fusion.  The source unified CAVS-OQR selections are frozen;
only the prediction source for the selected `current@640` action changes.

The experiment covers COCO and Oregon without retraining a learner, rerunning a
detector, changing a byte budget, or modifying the active unified experiment.

## Commands

- Test: `python -m unittest discover -s tests -v`
- Preflight: `python scripts/run_experiment.py --stage preflight`
- Run: `python scripts/run_pipeline.py`

## Project Structure

- `configs/protocol.json`: frozen sources, roles, budgets, bootstrap and resource limits.
- `scripts/core.py`: pure protocol and transport-invariance contracts.
- `scripts/run_experiment.py`: source verification, MAUR construction, evaluation and bootstrap.
- `scripts/run_pipeline.py`: resumable COCO-then-Oregon orchestration.
- `tests/`: unit and runner contract tests.
- `runs/`: experiment-local generated artifacts only.

## Testing Strategy

- Validate the exact two datasets, one ROI action, two fusion methods and three budgets.
- Validate every source CAVS-OQR selection key and action.
- Require MAUR and Reversible rows to have identical selected images, action counts,
  request bytes and KB/image.
- Re-evaluate Reversible and require it to reproduce the frozen source point row.
- Require COCO image bootstrap and Oregon event-block primary plus image sensitivity.
- Require exactly 3 + 6 = 9 paired-bootstrap jobs with 1,000 replicates each.

## Boundaries

- Always: use the completed unified experiment's exact CAVS-OQR selections.
- Always: use `current@640` ROI geometry, mapped ROI predictions and Edge predictions
  from the same detector cache as Reversible.
- Always: run with at most four low-priority CPU workers and one math-library thread.
- Never: retrain CAVS, select a threshold, rerun detector inference, or alter payload bytes.
- Never: modify, terminate or rewrite any source experiment or its active processes.
- Never: describe reused COCO val2017 or Oregon final as hidden/confirmatory evidence.

## Success Criteria

1. COCO and Oregon point rows complete for Fixed-byte and Fixed-call 10/25/40.
2. Transport invariance is exact for every MAUR/Reversible pair.
3. Fixed-call rows with zero ROI reproduce identical AP exactly.
4. Nine paired-bootstrap jobs finish and a Chinese report is written.
5. Source experiments remain read-only and the active unified run is unaffected.

## Open Questions

None.  Historical MAUR results remain reference-only because their selections and
training labels differ from this experiment.
