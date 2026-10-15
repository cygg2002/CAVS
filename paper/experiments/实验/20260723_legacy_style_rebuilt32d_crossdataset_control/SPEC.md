# Specification: Legacy-Style Rebuilt 32D Cross-Dataset Ranker Control

## Objective

Run one isolated exploratory paper-level control on COCO, ENA24, and Oregon
to compare three ranking formulations under the already-frozen detector,
split, evaluator, scheduler, and TriAction policies:

1. current 49D `SemanticRank`;
2. `Current-Controlled-32D`;
3. `Legacy-Style-Rebuilt-32D`.

The new ranker uses the historical 32D semantic-ranker *recipe*—a
ClassGainPrior anchor, 32D residual pairwise ranker, and fixed score scale
0.25—but refits every learned quantity on each dataset's current fit role.
It is not called `Legacy-Frozen-32D` on COCO or Oregon. ENA24's previously
completed frozen model is reported only as a separately labeled bridge
reference.

## Tech Stack

- Python 3.10-compatible existing evaluation scripts and frozen canonical
  JSON/CSV inputs.
- NumPy and the already-installed scikit-learn runtime; no new dependency.
- Current frozen 20260720 detector prediction/FPN caches; no detector
  training or inference rerun.

## Commands

```bash
python3 -m unittest discover -s tests -v
python3 scripts/run_dataset.py --dataset coco --stage preflight
python3 scripts/run_dataset.py --dataset ena24 --stage preflight
python3 scripts/run_dataset.py --dataset oregon --stage preflight
python3 scripts/run_pipeline.py --dataset DATASET
```

## Project Structure

```text
configs/protocol.json       Frozen ranker/matrix contract
scripts/evaluation_core.py  Pure validation and feature/ranker helpers
scripts/run_dataset.py      Dataset-isolated core/bootstrap runner
scripts/run_pipeline.py     Resumable core -> bootstrap pipeline
tests/                      Unit and runner contract tests
runs/{dataset}/             New outputs only
notes/                      Status and final Chinese report
```

## Code Style

```python
legacy_scores = legacy_anchor + score_scale * legacy_residual
```

Use explicit ranker names, stable image-ID alignment, finite-shape checks and
additive output files. Reuse the validated TriAction runner rather than
changing any source experiment.

## Testing Strategy

- Unit-test protocol cardinality and matrix counts for ten Rankers.
- Unit-test that Legacy-style scores are a 32D, finite, anchored residual with
  fixed scale 0.25.
- Keep all current action and fixed-byte contracts.
- Require current canonical SharedAction rows to reproduce at `1e-12`.
- Require input SHA-256 snapshots before/after both stages to match.
- Preflight every dataset before launching core work.

## Boundaries

### Always

- Write only inside this experiment directory.
- Refit PCA/anchor/residual weights on the current fit role only.
- Freeze score scale, hyperparameters, action policies and budgets before
  final evaluation.
- Keep `Current-Controlled-32D` and current 49D SemanticRank in the same
  matrix.
- Use lower-priority independent remote processes.

### Ask First

- Retraining any detector, creating a new FPN cache, changing a canonical
  split, or calling rebuilt COCO/Oregon models `Legacy-Frozen-32D`.

### Never

- Modify an existing canonical or active exploratory experiment.
- Use calibration/audit/final data to fit the rebuilt ranker.
- Tune score scale or hyperparameters based on final results.
- Merge ENA24's true frozen bridge result into cross-dataset averages.

## Success Criteria

- All three preflights and unit tests pass.
- The primary matrix contains ten Rankers (Random + nine non-random), 429
  rows per dataset; per-ranker ablation contains 135 rows; bootstrap contains
  27 comparisons with 200 replicates each.
- `Legacy-Style-Rebuilt-32D` has a persisted, fit-only 32D model and produces
  valid scores for calibration and final roles on every dataset.
- Existing rankers' SharedAction values reproduce their canonical rows exactly
  within `1e-12`.
- A report compares the rebuilt Legacy-style ranker, Current-Controlled-32D
  and current 49D without claiming a cross-dataset frozen-model comparison.

## Open Questions

- None for this run. The historical ENA24 frozen model remains an appendix
  reference and is not retrained or substituted into the main matrix.
