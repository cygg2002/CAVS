# DCSB + Adaptive Feeding external completion

Run one dataset at a time from the repository root:

```bash
python paper/experiments/20260829_dcsb_adaptive_feeding_native_baselines_v1/scripts/run_experiment.py \
  --dataset coco --stage all
```

Valid datasets are `coco`, `oregon`, and `voc`; stages are `preflight`, `train`,
`point`, and `all`.  Outputs are isolated under `runs/<dataset>/`.

The native rows implement the external rules.  Rows containing
`BudgetMatched` are explicitly evaluator-side budget normalization and not
claimed as original-paper algorithms.
