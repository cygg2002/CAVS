# Spec: Three-Dataset Cost-Aware Submission Ablations

## Objective

Create an isolated, reproducible post-hoc experiment package that strengthens the
paper's cost-aware configuration-selection claim without modifying the frozen
20260723 or 20260725 experiments.

This package answers four questions using only already-frozen tabular outputs:

1. Is one Ranker-family/action-policy pair feasible across all three datasets?
2. What is the final diagnostic cost of the least-violating shared pair?
3. What changes when the empirical gate, cost objective, or learned selector is
   removed?
4. Which requested experiments cannot be executed locally without independent
   audit assets, new data, external implementations, or real devices?

All final-based comparisons are explicitly post-hoc diagnostics. They are not
confirmatory experiments and cannot change any frozen configuration.

## Inputs

- `20260725_three_dataset_costaware_selector_riskgate/runs/*/candidate_table.csv`
- `20260725_three_dataset_costaware_selector_riskgate/runs/ena24/strict_candidate_table.csv`
- `20260725_three_dataset_costaware_selector_riskgate/runs/*/final_results.csv`
- `20260725_three_dataset_costaware_selector_riskgate/runs/*/provisional_choices.json`
- `20260725_three_dataset_costaware_selector_riskgate/runs/ena24/strict_provisional_choices.json`
- `20260723_three_dataset_nineranker_triaction_paper_exploration/runs/{coco,oregon}/results.csv`
- `20260725_three_dataset_costaware_selector_riskgate/runs/ena24/final_candidate_results.csv`

Every input is read-only and recorded with SHA-256 provenance.

## Shared-Policy Definition

"Shared" means the same Ranker family and action-policy name in all datasets.
Dataset-specific fitted weights, action thresholds, online thresholds, and
fixed-byte lambdas remain those frozen on each dataset's selector data. This is
a shared policy-family ablation, not one globally shared parameter vector.

For every registered work point:

- Compute each dataset's best `SharedAction` selector reference.
- A pair is strictly feasible only if its selector AP is at least
  `reference AP - 0.001` in every dataset.
- For Fixed-call and Online, rank strictly feasible pairs by mean normalized
  request cost, then worst AP delta, mean AP delta, and frozen tie-break order.
- For Fixed-byte, rank strictly feasible pairs by mean AP delta, worst AP
  delta, mean normalized request cost, and frozen tie-break order.
- If no pair is strictly feasible, do not relax epsilon. Select a diagnostic
  least-violating pair by: fewest violating datasets, highest worst AP delta,
  highest mean AP delta, lowest mean normalized request cost, then frozen
  tie-break order.

Request cost is normalized by each dataset's reference KB/image so Oregon's
larger source images do not dominate the cross-dataset objective.

## Lesion Variants

- `full_protocol`: the already-frozen 20260725 result.
- `no_gate`: deploy the selector-only provisional choice.
- `no_cost_objective`: choose selector AP maximum over all 24 candidates,
  ignoring request bytes.
- `fixed_confidence_shared`: use `Confidence + SharedAction` everywhere.
- `dataset_reference`: use each work point's best selector `SharedAction` row.
- `final_oracle`: final-only best AP candidate, diagnostic only.

Each variant is evaluated against the final result of `dataset_reference`.
An epsilon violation is `variant final AP - reference final AP < -0.001`.

## Commands

```bash
python3 -m unittest discover -s tests -v
python3 scripts/run_analysis.py
python3 scripts/run_bootstrap_workpoint.py --dataset coco --mode fixed_call --budget 0.10
```

## Project Structure

```text
configs/protocol.json        Frozen analysis contract
scripts/analysis_core.py     Pure selection and joining logic
scripts/run_analysis.py      Read-only input loader and report writer
scripts/run_bootstrap_workpoint.py  One-work-point bootstrap shard runner
tests/test_analysis_core.py  Unit tests for selection contracts
runs/                        Generated CSV/JSON outputs
notes/results_zh.md          Generated Chinese result summary
```

## Code Style

- Python standard library only.
- Pure functions accept rows and return rows; filesystem access stays in the
  runner.
- Deterministic lexical/tie-break ordering; no implicit dictionary ordering.
- ASCII in code; Chinese is limited to documentation and the final report.

## Testing Strategy

- Unit tests cover strict feasibility, no-epsilon-relaxation fallback,
  normalized cost, fixed-byte objective, and lesion selection.
- Integration checks require exactly 240 selector candidates per dataset,
  exactly 24 final candidates per registered work point, and exactly 30 rows
  per cross-dataset comparison.
- Provenance hashes are recomputed before and after analysis.
- Parallel bootstrap shards may only write their unique raw checkpoint and
  per-work-point provenance. The original serial runner remains the sole writer
  of `sensitivity.csv`, dataset-level `provenance.json`, and `COMPLETED`.

## Boundaries

### Always

- Keep all outputs under this new experiment directory.
- Mark final-based output as post-hoc and diagnostic-only.
- Preserve the registered epsilon and work points.
- Record missing assets and blocked experiments explicitly.

### Never

- Modify any 20260720-20260725 experiment.
- Use final AP to choose the strict shared pair or any deployable method.
- Treat an existing final table as a new hidden test.
- Reuse fit data as an independent gate.
- Claim that more bootstrap replicates increase statistical power.
- Label estimated latency as measured latency.

## Success Criteria

- Existing source hashes are unchanged.
- Tests fail before implementation and pass after implementation.
- Ten shared-policy work points and 30 dataset-level final diagnostics exist.
- Lesion output contains six variants for all 30 dataset/work-point pairs.
- Strictly infeasible work points remain marked infeasible.
- The report distinguishes completed diagnostics from blocked experiments.
- Parallel and serial execution produce the same registered action digest,
  seed, replicate prefixes, and raw checkpoint path for every work point.

## Open Questions

- Replaying 1000/2000-replicate audit bootstrap needs remote per-image audit
  predictions and actions that are not present in the local frozen package.
- A faithful external baseline needs an implementation choice and a new frozen
  protocol.
- Real system measurements need device/GPU availability and network shaping.
