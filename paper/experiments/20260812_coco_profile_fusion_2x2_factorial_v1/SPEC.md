# Spec: COCO Current/Historical × Reversible/MAUR 2×2 factorial

## Objective

Separate the effects of ROI expansion Profile, ROI fusion, and their interaction
under one canonical CurrentActionValue/CAVS protocol. The four cells are:

| Method | Profile | Fusion |
|---|---|---|
| CAVS-Current-Reversible | current@640 | Reversible |
| CAVS-Current-MAUR | current@640 | MAUR |
| CAVS-Historical-Reversible | historical_cost@640 | Reversible |
| CAVS-Historical-MAUR | historical_cost@640 | MAUR |

The first cell is an exact frozen anchor from the completed current@640 experiment.
The other three cells are retrained from their own contextual AP50-95 rewards with
the same model architecture, folds, seeds and scheduler.

## Commands

- Test: `python -m unittest discover -s tests -v`
- Gate: `python scripts/run_experiment.py --stage gate`
- Full pipeline: `python scripts/run_pipeline.py`

## Project Structure

- `configs/protocol.json`: exact split, sources, cells, gate and resource contract.
- `scripts/core.py`: pure factorial, gate and job-matrix contracts.
- `scripts/build_actions.py`: same-source MAUR action construction.
- `scripts/build_rewards.py`: fold-contained MAUR contextual rewards.
- `scripts/run_experiment.py`: teacher gate, training, evaluation and bootstrap.
- `scripts/run_pipeline.py`: prerequisite wait and resumable orchestration.
- `tests/`: protocol and behavior tests.
- `runs/`: experiment-local artifacts only.

## Testing Strategy

- Validate the exact 2×2 cell matrix and canonical COCO populations.
- Validate Current-Reversible is the only reused learner and is hash-bound.
- Validate exactly 54 new training jobs: 3 cells × 3 seeds × (5 folds + full).
- Validate all four cells use identical learner hyperparameters and base actions.
- Validate MAUR uses the same current/historical geometry, mapped prediction and
  payload as its Reversible counterpart.
- Validate 72 point rows and 15 paired-bootstrap contrasts.
- Validate final is never used for rewards, training, gating or model selection.

## Boundaries

- Always: canonical fit=9,678 / calibration=571 / val2017=5,000; Cloud model
  SHA `9fdd44...3e92`; scale=640; three seeds; frozen five folds; exact-capacity
  Fixed-byte 10/25/40 primary and Fixed-call auxiliary.
- Default: wait for the active unified and same-selection fusion experiments before
  consuming CPU. The user-authorized `--start-now` override may use otherwise idle
  CPUs concurrently; it remains low priority with bounded workers and one math thread.
- Always: use calibration-only GT-aware Teacher diagnostics for the pretraining gate.
- Never: modify a source experiment, rerun detector inference, change ROI geometry,
  tune on val2017, or describe this public-val reuse as hidden confirmation.

## Gate

For each non-anchor cell, compare its calibration GT-aware Fixed-byte Teacher
against Current-Reversible. A cell passes when at least two budgets improve by
at least 0.001 AP and the remaining budget is no worse than -0.001 AP. If any
new cell passes, train all three new cells so the factorial remains complete.
If none passes, stop before learner training and write a negative diagnostic report.

## Success Criteria

1. Both MAUR actions reproduce the exact paired Profile geometry and payload set.
2. Gate contains all four cells and three budgets without accessing final.
3. If opened, 54/54 training jobs complete and all four cells are evaluated once.
4. Fifteen 1,000-replicate paired bootstrap contrasts finish:
   two Fusion effects, two Profile effects and one factorial interaction per budget.
5. Source experiments remain byte-for-byte read-only.

## Open Questions

None. Historical published rows remain reference-only and cannot replace the
new Historical-MAUR cell.
