# Spec: COCO Relational Action-Value OOF Diagnostic v1

## Objective

Run one COCO-only, cache-only 2x2 diagnostic that isolates the two remaining
ActionValue hypotheses:

1. Does mean/max pooling discard useful relations between Edge detections?
2. Does reward regression plus within-image action ordering need an explicit
   cross-image/action ranking objective?

The complete candidate is `RelActionRank`: a permutation-equivariant detection
set encoder followed by action-conditioned value prediction and a cross-image
pairwise ranking loss. It is evaluated only for Edge/Original/Q90; ROI, risk
quantiles, Selection Context, and new detector inference are outside scope.

## Frozen data and actions

- Dataset: COCO canonical train2017 roles only.
- Fit: exactly 9,678 images.
- Calibration: exactly 571 disjoint images.
- OOF: five deterministic folds and three frozen seeds.
- Actions: Edge fallback plus Original and Q90 candidates.
- Primary budget: Fixed-byte at 10%, 25%, and 40% of all-Original bytes.
- Secondary budget: Fixed-call at 10%, 25%, and 40%.
- `val2017`, audit, final, and registered-final are forbidden.

## Fold-contained contextual rewards

For each fit target, its 32 context images are sampled only from the target's
own deterministic OOF fold. Therefore, when any other fold is held out, every
training target and all IDs used in its reward context exclude that holdout.
Original and Q90 receive separate AP50-95 contextual rewards. Calibration
contexts contain calibration IDs only and are never used for training or model
selection.

This replaces the earlier transductive reward cache whose context could cross
OOF folds. Rewards are generated from frozen predictions and GT only; no image
or detector inference is performed.

## Methods

| Method | Detection encoder | Objective |
|---|---|---|
| `CurrentActionValue` | frozen mean/max pooling design | AP gain regression + within-image action ordering |
| `GlobalRank` | mean/max pooling | baseline objective + cross-image/action pairwise ranking |
| `RelSetValue` | two-layer self-attention set encoder without positional indices | baseline objective |
| `RelActionRank` | relational set encoder | baseline objective + cross-image/action pairwise ranking |

All four methods use the same cost-stripped scene/action inputs, optimizer,
hidden dimensions, folds, seeds, rewards, and scheduler. Payload never enters
the learned value path. Exact payload enters only the Fixed-byte scheduler.

Frozen calibration references may be reported for DualHead, SetValue,
SemanticRank, and the strongest ORIC/MORIC variant, but they cannot choose a
model, epoch, loss weight, or protocol.

## Commands

```bash
python -m unittest discover \
  -s paper/experiments/20260806_coco_relational_actionvalue_oof_v1/tests -v

python paper/experiments/20260806_coco_relational_actionvalue_oof_v1/scripts/run_experiment.py \
  --stage all
```

## Project structure

```text
configs/protocol.json  frozen data, model, reward, gate, and CPU contract
scripts/core.py        pure folds, isolation, model, loss, metrics, and gate
scripts/run_experiment.py  resumable reward, OOF, calibration, and reporting
tests/                 unit and source-boundary tests written before logic
runs/                  generated checkpoints, scores, selections, and reports
```

## Code style

Use typed, deterministic, experiment-local Python. Reuse completed canonical
loaders/evaluators without modifying their sources.

```python
assert set(context_ids) <= set(folds[target_fold])
assert not set(context_ids) & set(heldout_ids)
```

## Testing strategy

- Unit tests: protocol isolation, exact fold coverage, fold-contained reward
  contexts, No-ROI enforcement, relation-encoder permutation invariance,
  cross-image loss behavior, rank metrics, capacity, and success gates.
- Integration preflight: source completion, role populations, action sources,
  ID alignment, hashes, and forbidden-role flags.
- Training smoke: every model performs forward/backward on synthetic tensors;
  all output scores remain finite and cost-free.
- Remote preflight and tests must pass before background launch.

## Boundaries

- Always: write only inside this experiment; hash sources; checkpoint each
  stage; use fold-contained rewards; report negative outcomes.
- Ask first: change roles, actions, context size, folds, seeds, loss weights,
  architecture dimensions, budgets, or gates.
- Never: access val2017/audit/final/registered-final; use calibration for
  training or parameter selection; modify completed source experiments; call
  this diagnostic a paper-final result.

## Success criteria

1. Every fit image receives one OOF prediction per method and seed.
2. Every fit training reward excludes the corresponding held-out fold.
3. No selection contains ROI and every Fixed-byte capacity is respected with
   at least 98% utilization.
4. `RelActionRank` passes OOF only if, relative to `CurrentActionValue`, it:
   - improves at least two Fixed-byte points by at least 0.001 AP50-95;
   - is no worse than -0.001 at the remaining Fixed-byte point;
   - is no worse than -0.001 at every Fixed-call point; and
   - has the same positive direction in at least two of three seeds at each
     required gain point.
5. Calibration is evaluated once only if the OOF gate passes; it must satisfy
   the same AP/capacity gate to open a future paper protocol.

## Open questions

None. The user approved this exact COCO-only 2x2 diagnostic and background CPU
execution.
