# Spec: COCO Factorized Global Action Rank OOF Diagnostic v1

## Objective

Run one final COCO-only diagnostic to test whether the high-budget failure of a
single ActionValue score is caused by coupling image eligibility and action
choice. The new `FactorizedGlobalActionRank` predicts:

1. a global image value `G_i` for whether any cloud action is useful; and
2. an action advantage `A_i,a` for Original versus Q90.

The deployed value is `V_i,a = G_i + A_i,a`. Payload is forbidden from the
learned value path and is used only by the exact Fixed-byte scheduler.

## Frozen protocol

- Dataset: COCO canonical train2017 roles only.
- Fit: exactly 9,678 images; five OOF folds; three frozen seeds.
- Calibration: exactly 571 disjoint images, evaluated once only after OOF pass.
- Actions: Edge fallback plus Original and Q90; ROI is forbidden.
- Primary: Fixed-byte 10%, 25%, 40%; secondary: Fixed-call 10%, 25%, 40%.
- `val2017`, audit, final, and registered final are forbidden.
- Reuse the prior experiment's hashed fold-contained rewards and frozen OOF
  scores for `CurrentActionValue`, `GlobalRank`, and `RelActionRank`.
- Train only `FactorizedGlobalActionRank` in this diagnostic.

## Learning objective

For normalized contextual rewards `r_i,a` including the Edge zero baseline:

```python
global_target = max(r_edge, r_original, r_q90)
advantage_target[a] = r_i_a - global_target
value[a] = global_prediction + advantage_prediction[a]
```

The four equally weighted losses are global reward regression, cross-image
global ranking, cloud-action advantage regression, and within-image action
ranking. The factorized model uses the stable mean/max detection-set encoder;
no new relation layers, risk head, ROI, or selection context are added.

## Commands

```bash
python -m unittest discover \
  -s paper/experiments/20260806_coco_factorized_global_actionrank_oof_v1/tests -v

python paper/experiments/20260806_coco_factorized_global_actionrank_oof_v1/scripts/run_experiment.py \
  --stage all
```

## Project structure

```text
configs/protocol.json  frozen source, model, loss, budget, and gate contract
scripts/core.py        factorized targets, model, loss, result validation
scripts/run_experiment.py  source binding, OOF jobs, evaluation, reporting
tests/                 unit and protocol tests
runs/                  resumable generated artifacts
```

## Testing strategy

- Unit-test factorized target reconstruction, cloud-only action advantage,
  cross-image global ordering, cost stripping, No-ROI behavior, and exact
  result matrix.
- Verify reused reward/score hashes and image-ID alignment before training.
- Run local and remote tests before the background launch.
- Evaluate calibration only if the frozen OOF gate passes.

## Boundaries

- Always: reuse frozen reward and baseline OOF artifacts read-only; train only
  the new model; record hashes; preserve exact budgets and evaluator.
- Ask first: change folds, seeds, loss weights, actions, budgets, or gate.
- Never: modify the source experiment, use payload in a learned head, access
  val2017/audit/final, or tune on calibration.

## Success criteria

Relative to frozen `CurrentActionValue`, `FactorizedGlobalActionRank` must:

1. improve at least two Fixed-byte budgets by at least 0.001 AP50-95;
2. be no worse than -0.001 at the remaining Fixed-byte budget;
3. be no worse than -0.001 at all Fixed-call budgets;
4. have at least two supporting seeds at each required gain point;
5. use at least 98% of every Fixed-byte capacity and select no ROI.

Only an OOF pass permits one frozen calibration evaluation. Calibration must
pass the same gate to open a future paper protocol.

## Open questions

None. The user approved this as the final bounded COCO diagnostic.
