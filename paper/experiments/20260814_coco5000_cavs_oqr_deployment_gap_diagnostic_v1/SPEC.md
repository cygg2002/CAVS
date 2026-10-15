# COCO5000 CAVS-OQR deployment-gap diagnostic

## Objective

Explain the AP50-95 gap between the frozen cached-paper CAVS-Current-MAUR result
and the 5,000-image Mac-to-RTX4090 system replay without retraining or changing
any selection after evaluation.

## Inputs

- Frozen real-system plans, candidates, and mixed predictions from
  `20260814_coco5000_cavs_oqr_end_to_end_v1/runs/formal`.
- Frozen paper selections and Current-MAUR action caches on the experiment server.
- The canonical COCO val2017 annotation file and all 5,000 image ids.

## Diagnostics

1. Re-evaluate the published paper selection from cached action predictions and
   require exact agreement with the recorded paper AP.
2. Evaluate the real selection with all cached actions to isolate selection/cost
   effects from execution effects.
3. Re-schedule the real scores first with the native-file budget denominator and
   then with the full paper per-action byte costs.
4. Evaluate all 16 real/cache action-source combinations on the same real
   selection and compute exact Shapley attribution for Edge, Original, Q90, and
   Current@640+MAUR.
5. Report action counts, byte totals, AP decomposition, and prediction-count
   agreement for each action.
6. Materialize byte-identical OpenCV Q90 images and run them through the frozen
   paper validator to split the Q90 effect into codec and realtime-inference terms.

## Commands

```bash
pytest -q tests/test_diagnostic.py
nice -n 10 python scripts/run_diagnostic.py --output runs/formal
nice -n 10 python scripts/run_q90_cross.py --output runs/q90_cross
```

## Boundaries

- Always: use frozen selections and prediction artifacts; verify the paper point
  before interpreting any delta; keep all outputs in this experiment directory.
- Never: retrain a learner, rerun a detector, access GT during scheduling, modify
  source experiments, or present path-dependent selection effects as causal.

## Success Criteria

- Cached paper replay matches all three recorded AP50-95 values within `1e-6`.
- Real mixed replay matches the completed system report within `1e-6`.
- The published-to-real AP difference is numerically reconciled into selection/
  cost and execution terms.
- Execution attribution sums to the cached-real minus real-real AP difference
  within floating-point tolerance.
- The Q90 crossover reproduces the cached and real-Q90 hybrid anchors and its
  codec plus inference terms reconcile exactly.
