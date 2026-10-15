# Spec: AFPD Same-Run Baseline Bootstrap v1

## Objective

Run a read-only, post-hoc paired-bootstrap sidecar for the two attribution
comparisons missing from the ongoing AFPD full-val experiment:

1. `AFPD-OQR` versus the independently retrained `Current-OQ`;
2. `AFPD-OQR` versus the independently retrained `Current-OQR`.

The experiment reports Fixed-byte 10%/25%/40% only. It does not train a model,
reschedule an action, access a new split, or select a replacement method.

## Commands

```bash
python -m unittest discover \
  -s paper/experiments/20260810_coco_afpd_same_run_baseline_bootstrap_v1/tests -v

python paper/experiments/20260810_coco_afpd_same_run_baseline_bootstrap_v1/scripts/run_experiment.py \
  --stage preflight

python paper/experiments/20260810_coco_afpd_same_run_baseline_bootstrap_v1/scripts/run_experiment.py \
  --wait-for-parent

# Explicit resource-reviewed override: run concurrently with the parent.
python paper/experiments/20260810_coco_afpd_same_run_baseline_bootstrap_v1/scripts/run_experiment.py \
  --allow-concurrent-parent
```

## Project structure

```text
SPEC.md
PLAN.md
configs/protocol.json
scripts/core.py
scripts/run_experiment.py
tests/test_core.py
runs/
```

## Code style

Use experiment-local deterministic Python and reuse the completed parent's
prediction/evaluator interfaces read-only.

```python
for comparison in comparison_specs(protocol):
    run_paired_bootstrap(comparison)
```

## Testing strategy

- Unit-test the exact six-comparison matrix and protocol boundaries.
- Unit-test selection ID/action validation.
- Remote preflight must fingerprint the parent code and final selections.
- Execution waits for the parent's main bootstrap completion marker by default.
  An explicit `--allow-concurrent-parent` override is permitted only after a
  resource check and is recorded in the output.

## Boundaries

- Always: reuse the same-run `final/evaluation/selections.json`, the canonical
  5,000-image ground truth, and 1,000 paired image resamples.
- Ask first: adding another method, mode, budget, or resampling unit.
- Never: modify the parent experiment, reuse the older frozen ActionValue
  selection as either comparator, tune on val2017, or call the result
  confirmatory/hidden-final evidence.

## Success criteria

1. Preflight verifies the parent full-val is complete. It either verifies the
   parent main bootstrap is complete or records an explicit concurrent override.
2. Exactly six unique comparisons are produced, each with 1,000 replicates.
3. Candidate and comparator selections come from the same independent-retrain
   run, and every selected ID belongs to the same 5,000-image final population.
4. Source hashes are recorded; parent files are unchanged.
5. Output states `confirmatory=false`, `final_is_hidden=false`, and
   `final_used_for_selection=false`.

## Open questions

None. This is a descriptive attribution bootstrap on an already observed,
reused val2017 final.
