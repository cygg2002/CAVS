# Spec: COCO Budget-Set SharedAction + CostEligible Diagnostic

## Objective

Test whether our own deployable detection-set encoder plus budget-boundary
training and CostEligible multi-action scheduling can outperform retrained
ORIC/MORIC controls on a newly sealed COCO train2017 holdout.

This is a mechanism diagnostic, not a paper final. It must not read the already
consumed COCO val2017 final from
`20260804_coco_budget_contextual_ranker_v2`.

Primary comparisons:

1. `ORIC-MLP-Full`: paper-style top-25 feature, raw contextual AP50-95 reward,
   Full action only.
2. `MORIC-MLP-Full`: the same feature, CDF target, Full action only.
3. `BudgetSet-Full`: our set encoder and exact budget-boundary objective, Full
   action only.
4. `BudgetSet-SharedAction`: the same learned model, jointly choosing Edge,
   Full, ROI or Pair through CostEligible scheduling.

## Assumptions

- The completed GAIR action cache and CA_AORIC AP50-95 reward rows are immutable
  read-only inputs; no detector inference is required.
- Only edge detections, image geometry, candidate geometry, action type and
  resource cost are deployable model inputs. `miss_summary`, `frame_anchors`,
  learned anchors, cached cloud outputs and ground truth are forbidden inputs.
- A deterministic SHA-256 image-id split freezes 4,000 training images and
  1,000 sealed holdout images before model fitting.
- Existing CA_AORIC rewards used context images drawn from the full 5,000-image
  cache. This makes the run a transductive reward-label diagnostic; the holdout
  target rewards are never used for training or model selection, and reported
  AP is evaluated directly from frozen predictions.

## Tech Stack

- Python 3, NumPy, PyTorch, standard-library `unittest`.
- Read-only GAIR-v5 sharded action cache and CA_AORIC reward JSONL.
- Existing COCO metric backend and multiple-choice resource scheduler.

No new dependency is added and no existing experiment is modified.

## Commands

```bash
python -m unittest discover \
  -s paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1/tests -v

python paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1/scripts/run_experiment.py \
  --stage prepare
python paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1/scripts/run_experiment.py \
  --stage train
python paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1/scripts/run_experiment.py \
  --stage holdout
```

## Project Structure

```text
configs/protocol.json       Frozen split, model and success thresholds
scripts/experiment_core.py  Pure split, feature, loss and selection contracts
scripts/run_experiment.py   Gated prepare/train/holdout runner
tests/                      Unit and stage-isolation tests
runs/                       New artifacts only
notes/results_zh.md         Result interpretation
```

## Code Style

Use small typed functions and deterministic stable ordering:

```python
order = np.argsort(-np.asarray(scores), kind="mergesort")
```

## Model and Objective

- Up to 64 highest-confidence Edge boxes are represented as an unordered set.
- Each box uses normalized `cx, cy, w, h, area, confidence` and a learned class
  embedding. Mean/max pooling produces a permutation-invariant image state.
- Each deployable candidate uses action geometry, type, byte/GFLOP/E2E cost and
  the requested budget. Teacher-only anchors are excluded.
- The model predicts contextual AP50-95 reward for every candidate action.
- Training combines action-reward regression, within-image action ordering and
  cross-image pairs drawn across the exact 10%/25%/40% teacher budget boundary.
- `CostEligible` uses the existing multiple-choice Lagrangian scheduler under
  the same absolute Full-upload capacity for all methods.

## Testing Strategy

- Unit-test deterministic disjoint splits and sealed-holdout contracts.
- Unit-test permutation-invariant box encoding and forbidden-field exclusion.
- Unit-test exact boundary pairs, action masks and fixed-call selection.
- Unit-test CostEligible never exceeds its byte capacity.
- Synthetic training smoke must rank the correct image/action above distractors.
- Run existing CA_AORIC and GAIR selector tests as regressions.

## Boundaries

- Always: train and choose epochs only on the 4,000 training images; capture
  source fingerprints; report Full-only and SharedAction separately.
- Ask first: changing the cache, detector, reward source, split seed, budgets or
  success thresholds after holdout access.
- Never: read val2017 final; train on holdout rewards; use teacher-only cache
  fields as inputs; overwrite completed runs; tune after holdout evaluation.

## Success Criteria

Primary fixed-byte go/no-go on the sealed holdout:

- `BudgetSet-SharedAction` exceeds the better of retrained ORIC/MORIC by at
  least 0.002 AP at two of three budgets.
- It loses no more than 0.001 AP at the remaining budget.
- Its upload bytes never exceed the common resource capacity.

Secondary fixed-call target:

- At 10%, upload KB/image is no greater than the better external control and AP
  is no lower.

The result must be reported as diagnostic even if all gates pass.

## Open Questions

None. Hyperparameters and split are frozen in `configs/protocol.json` before
holdout access.
