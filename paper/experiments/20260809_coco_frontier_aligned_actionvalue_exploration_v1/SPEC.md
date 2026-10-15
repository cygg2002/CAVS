# Spec: COCO Frontier-Aligned ActionValue Exploration v1

## Objective

Determine whether the remaining gap of COCO ActionValue comes from a mismatch
between its per-action reward/ranking loss and the multiple-choice byte-budget
scheduler used at deployment.  Explore a learner that represents the action
Pareto frontier over Edge, Original, Q90, and optional ROI, and is trained on
counterfactual shadow-price decisions rather than only scalar reward error.

This is a development diagnostic, not a paper-final experiment.  A literature
audit and an empirical failure audit precede implementation.  Novelty claims
must be phrased as evidence-bounded: absence from the searched corpus is not
proof that no prior paper exists.

## Assumptions

1. Canonical COCO train2017 fit=9,678 and calibration=571 remain the only roles.
2. Existing detector predictions, fold-contained AP50-95 rewards, payloads, and
   folds are read-only; no detector inference is required.
3. The currently running full COCO/Oregon benchmark remains untouched.
4. Fixed-byte 10%/25%/40% is primary; Fixed-call is a safety check.
5. ROI is optional and must compete with Edge/Original/Q90 rather than be forced.

## Methods under consideration

- `CurrentActionValue-OQ`: frozen OOF baseline.
- `CurrentActionValue-OQR`: frozen ROI-enabled baseline where comparable.
- `FrontierAligned-OQ`: the same deployable inputs and base encoder, augmented
  only with a shadow-price decision loss.
- `FrontierAligned-OQR`: the same objective with ROI as an optional action.
- `DeltaFrontier-OQR`: only if the first diagnostic supports it, add an
  antisymmetric pairwise action-delta head and consistency loss.
- `SetValuedFrontier-OQR`: only if ensemble disagreement is informative, retain
  a Pareto-safe action set instead of forcing an uncertain unique action.

The candidate list is staged; later candidates are not implemented merely
because they are listed.

## Tech stack and versions

- Python and PyTorch from the existing `yolo26` environment.
- Existing experiment-local COCO loaders, evaluator, reward cache, and GAIR
  multiple-choice Lagrangian scheduler.
- CPU-only training with one PyTorch thread per process.
- No new dependency.

## Commands

```bash
python -m unittest discover \
  -s paper/experiments/20260809_coco_frontier_aligned_actionvalue_exploration_v1/tests -v

python paper/experiments/20260809_coco_frontier_aligned_actionvalue_exploration_v1/scripts/run_experiment.py \
  --stage diagnose

python paper/experiments/20260809_coco_frontier_aligned_actionvalue_exploration_v1/scripts/run_experiment.py \
  --stage screen

python paper/experiments/20260809_coco_frontier_aligned_actionvalue_exploration_v1/scripts/run_experiment.py \
  --stage confirm
```

## Project structure

```text
SPEC.md                  frozen scope and scientific boundaries
PLAN.md                  staged tasks and go/no-go rules
configs/protocol.json    exact roles, actions, seeds, budgets, and gates
research/                auditable literature-search state and synthesis
scripts/core.py          pure frontier targets, losses, and gate contracts
scripts/run_experiment.py resumable diagnosis/training/evaluation runner
tests/                   unit and protocol tests
runs/                    generated development artifacts only
```

## Code style

Use deterministic, typed, experiment-local Python and import completed source
implementations read-only.

```python
target_utility = reward - shadow_price * normalized_payload
predicted_utility = value - shadow_price * normalized_payload
```

## Testing strategy

- Unit tests first for shadow-price targets, boundary weighting, ROI masking,
  cost-free value inputs, permutation invariance, and budget-capacity checks.
- Integration preflight for exact roles, fold isolation, source completion,
  score/reward/action alignment, and forbidden-role flags.
- One-seed five-fold OOF screening before any three-seed confirmation.
- Frozen calibration is evaluated at most once and never selects a model.

## Boundaries

- Always: distinguish observed fact, mechanistic inference, and judgment;
  preserve negative results; checkpoint stages; keep final roles sealed.
- Ask first: change canonical populations or run new detector inference.
- Never: modify completed experiments; use calibration/val2017/final for tuning;
  call an exploratory win confirmatory; claim exhaustive novelty from search.

## Success criteria

1. The diagnostic quantifies selection-boundary regret separately from reward
   regression, image membership, and within-image action error.
2. The literature record identifies adjacent decision-focused, ranking,
   adaptive inference, and edge-cloud scheduling work with direct citations.
3. A screened candidate advances only if one-seed OOF Fixed-byte improves at
   least two budgets by >=0.001 AP and the third is >=-0.001, with capacity
   utilization >=98% and no Fixed-call loss below -0.001.
4. A three-seed candidate passes only if its score ensemble satisfies the same
   point gate and at least two of the three Fixed-byte budgets each have >=2/3
   individual seeds with delta AP >=0.001.  Individual-seed support is measured
   against a seed-matched OQ CurrentActionValue model.  Calibration remains
   sealed until this OOF confirmation passes.
5. ROI is retained only if it improves the OQ counterpart at >=1 budget without
   violating the other gates; otherwise the method remains OQ.
6. No audit, val2017, final, or registered-final data is accessed.

## Open questions

- Whether the empirical mismatch is mostly global membership, action rivalry,
  or near-boundary value calibration.
- Whether ROI provides a learnable Pareto segment on COCO after decision-aligned
  training, despite its previously small oracle tail.
