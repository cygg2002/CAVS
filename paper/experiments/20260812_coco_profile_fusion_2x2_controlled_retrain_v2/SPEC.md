# Spec: COCO controlled Current/Historical × Reversible/MAUR 2×2 retraining

## Objective

Run a strictly controlled four-cell factorial experiment in which ROI Profile
and Fusion are the only factors allowed to change. Unlike the exploratory v1,
all four cells are retrained; no learner checkpoint or score is reused.

| Method | Profile | Fusion |
|---|---|---|
| CAVS-Current-Reversible | current@640 | Reversible |
| CAVS-Current-MAUR | current@640 | MAUR |
| CAVS-Historical-Reversible | historical_cost@640 | Reversible |
| CAVS-Historical-MAUR | historical_cost@640 | MAUR |

## Commands

- Test: `python -m unittest discover -s tests -v`
- Preflight: `python scripts/run_experiment.py --stage preflight`
- Run: `python scripts/run_pipeline.py --start-now`

## Project Structure

- `configs/protocol.json`: frozen factorial and resource contract.
- `scripts/build_rewards.py`: one shared fold/context manifest and all six
  Original/Q90/Profile×Fusion rewards.
- `scripts/run_experiment.py`: diagnostics, four-cell retraining, evaluation,
  paired bootstrap and report.
- `scripts/run_pipeline.py`: resumable low-priority orchestration.
- `tests/`: protocol and causal-control regression tests.
- `runs/`: experiment-local outputs only.

## Testing Strategy

- Require all four cells to declare `reused_anchor=false`.
- Require 72 jobs: four cells × three seeds × (five folds + full fit).
- Require one identical fold hash, context hash and inner-split algorithm for all cells.
- Require six rewards per image: Original, Q90 and four Profile×Fusion actions.
- Require paired Profile variants to preserve ROI geometry and payload exactly.
- Require 72 point rows and 15 paired-bootstrap contrasts.

## Boundaries

- Always: COCO fit=9,678 / calibration=571 / val2017=5,000; pinned canonical
  Cloud model; scale=640; three seeds; five folds; exact-capacity scheduler.
- Always: reuse the already generated, hash-bound FP16 OQR12 Reversible and
  paired MAUR prediction caches; detector inference is forbidden.
- Always: generate every cell's contextual reward with one shared folds.json
  and contexts.json and train every cell with the same adapter.
- Always: run with `nice=15`, one math thread per worker and no more than eight workers.
- Never: reuse the old Current-Reversible checkpoint, tune on val2017, modify or
  stop another experiment, or describe public-val reuse as hidden confirmation.

## Success Criteria

1. Preflight proves all four checkpoint families are new and source caches are unchanged.
2. Shared fold/context hashes cover all four cells and all reward rows.
3. Exactly 72/72 training jobs complete and all point rows are emitted once.
4. Fifteen 1,000-replicate paired-bootstrap contrasts complete.
5. Existing experiments remain read-only and continue running.

## Open Questions

None. This run corrects the causal-control defect identified in factorial v1.
