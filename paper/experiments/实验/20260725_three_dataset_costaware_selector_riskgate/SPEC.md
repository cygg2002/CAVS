# Specification: Three-Dataset Cost-Aware Selector and Risk Gate

## Objective

Build one isolated exploratory paper-level experiment that turns the completed
three-dataset nine-Ranker TriAction matrix into a deployable calibration-only
controller.

For each dataset, mode, and registered work point, the controller must:

1. select a reference from the eight deterministic `SharedAction` Rankers;
2. select one provisional `(Ranker, action policy)` using calibration only;
3. apply an uncertainty gate without reading final;
4. freeze either the provisional choice or the reference fallback;
5. evaluate the frozen choice on final exactly once by reusing the completed
   final rows and action selections.

The experiment must not modify or replace any completed canonical or
exploratory experiment.

## Assumptions

- The completed
  `20260723_three_dataset_nineranker_triaction_paper_exploration` artifacts are
  immutable inputs.
- `Random` remains a ninth reference Ranker but is excluded from automatic
  deployment selection because a random seed is not a reproducible learned
  policy.
- The automatic selector compares the three already calibrated action
  policies: `SharedAction`, `CostEligible-AllROI`, and
  `GBDT-Hierarchical`. It does not tune a new ROI threshold on final.
- Fixed-call automatic selection covers the registered primary budgets
  `10%/25%/40%`. Online covers `10%/25%/40%`; Fixed-byte covers
  `5%/10%/20%/40%`.
- AP tolerance remains `0.001`.
- COCO and Oregon use their existing frozen audit roles.
- ENA24 has no compatible independent audit role. Its 400-image calibration
  role is split before selection into near-duplicate-group-disjoint selector
  and gate partitions. All policy thresholds, Online frame thresholds, and
  Fixed-byte lambdas are recalibrated on the selector partition only.

## Frozen Candidate Contract

The eight selectable Rankers are:

1. `Confidence`
2. `ClassGainPrior`
3. `M2E`
4. `Confidence-Residual M2E`
5. `SemanticRank`
6. `MORIC-style`
7. `GEC-style`
8. `Current-Controlled-32D`

The policies are:

1. `SharedAction`
2. `CostEligible-AllROI`
3. `GBDT-Hierarchical`

For every dataset/mode/budget there are 24 deterministic candidates.

`GBDT-Hierarchical` means the already frozen dataset/mode shared threshold from
the 20260723 experiment. Fixed-byte candidates retain their already calibrated
candidate-specific lambda.

## Selection Algorithm

### Reference

At each dataset/mode/budget, choose the `SharedAction` Ranker with maximum
calibration Official AP. Break ties by lower request KB, then frozen Ranker
order.

### Provisional candidate

For Fixed-call and Online:

1. keep candidates whose calibration AP is at least
   `reference AP - 0.001`;
2. choose minimum request KB;
3. break ties by higher Official AP, simpler policy order, then frozen Ranker
   order.

For Fixed-byte:

1. keep candidates whose calibration AP is at least
   `reference AP - 0.001`;
2. choose maximum Official AP;
3. break ties by lower request KB, simpler policy order, then frozen Ranker
   order.

The simpler policy order is:
`SharedAction`, `GBDT-Hierarchical`, `CostEligible-AllROI`.

### Risk gate

Compare the provisional candidate against the frozen reference using 200
deterministic paired image-level bootstrap replicates.

- Gate passes when the two-sided 95% AP-delta CI lower bound is at least
  `-0.001`.
- If the actions are identical, the delta and CI are exactly zero.
- If the gate fails, freeze the reference unchanged.
- Do not try a second candidate after observing a gate failure.
- Do not use final to select, gate, or repair a choice.

Gate data:

- COCO: existing frozen 630-image audit role.
- Oregon: existing frozen 9,923-image audit role.
- ENA24: a pre-registered split of the 400-image calibration role.

ENA24 split contract:

1. compute a 256-bit grayscale dHash at 16x16 from each original image;
2. connect images with Hamming distance at most 6 and take union-find connected
   components;
3. assign whole components to selector/gate partitions using a deterministic
   rare-class and size-balancing objective with seed `20260725`;
4. target 200/200 images without splitting a component;
5. write image IDs, hashes, components, class balance, and source image hashes.

ENA24 does not contain reliable event/camera/location metadata. The split may
be described only as `near_duplicate_group_disjoint`; event/camera disjointness
remains unknown.

For Fixed-call and Online, a gate pass additionally requires the 95% upper
bound of request-KB delta to be at most zero. Fixed-byte keeps the byte-capacity
contract and gates AP only.

The 200-replicate intervals are exploratory pointwise intervals, not
simultaneous statistical guarantees.

### Final and oracle diagnostic

Map the frozen choice to the exact completed final result and action map.
Separately compute the final oracle over the same 24 candidates for diagnostic
regret only. The oracle must never alter the frozen choice.

## Global One-Configuration Ablation

For every mode/budget, select one `(Ranker, policy)` shared by all three
datasets using calibration only:

1. require point-estimate AP loss no worse than `-0.001` against each
   dataset-specific reference;
2. Fixed-call/Online: minimize mean normalized request KB across datasets;
3. Fixed-byte: maximize mean calibration AP;
4. break ties using the same deterministic orders.

This is an ablation that tests the “one universal learner/policy” hypothesis.
It cannot replace the dataset-adaptive primary controller.

## Real-System Measurement Gate

Before claiming measured E2E:

1. record the actual edge and cloud hosts, processors, accelerators, software
   versions, and model checkpoints;
2. verify that detector inference, encoding, transfer, decoding, and cloud
   inference can be timed separately;
3. verify that bandwidth/RTT conditions are measured or controlled;
4. never label a single-host estimate or formula as real end-to-end latency.

If a distinct edge/cloud path is unavailable, produce a machine-readable
feasibility report and do not fabricate real-system results.

## Inputs

Read-only source:

```text
paper/experiments/20260723_three_dataset_nineranker_triaction_paper_exploration/
  configs/protocol.json
  runs/{coco,ena24,oregon}/calibration_curve.csv
  runs/{coco,ena24,oregon}/threshold_choices.json
  runs/{coco,ena24,oregon}/results.csv
  runs/{coco,ena24,oregon}/selections.json
  runs/{coco,ena24,oregon}/models/
```

Canonical role manifests and detector/action caches remain read-only inputs.

## Commands

```bash
python3 -m unittest discover -s tests -v
```

```bash
python scripts/run_selector.py --dataset DATASET --stage preflight
python scripts/run_selector.py --dataset DATASET --stage calibrate
python scripts/run_selector.py --dataset DATASET --stage risk-gate
python scripts/run_selector.py --dataset DATASET --stage final
```

```bash
python scripts/summarize_cross_dataset.py
python scripts/audit_system_measurement.py
```

## Project Structure

```text
configs/protocol.json
scripts/selector_core.py
scripts/run_selector.py
scripts/summarize_cross_dataset.py
scripts/audit_system_measurement.py
tests/
runs/{dataset}/
notes/
```

## Testing Strategy

Tests are written before implementation and cover:

1. exact 24-candidate grid and Random exclusion;
2. deterministic reference selection;
3. Fixed-call/Online minimum-cost selection under AP tolerance;
4. Fixed-byte maximum-AP selection under AP tolerance;
5. ENA24 near-duplicate components never cross selector/gate;
6. ENA24 thresholds, Online frame thresholds, and Fixed-byte lambdas are fitted
   on selector only;
7. no second candidate after a failed risk gate;
8. identical-action bootstrap shortcut;
9. final rows cannot participate in selection;
10. source hashes are unchanged before and after every stage;
11. incomplete real-system metadata prevents a measured-E2E claim.

## Boundaries

### Always

- Write only inside this experiment directory.
- Hash every source artifact actually read.
- Preserve dataset/mode/budget identities exactly.
- Record provisional choice, reference, gate CI, fallback, final result, and
  oracle regret.
- Preserve negative results unchanged.

### Never

- Modify the 20260720–20260723 experiment directories.
- Read final before the frozen selection file is written and hashed.
- Select a method using final AP.
- Recalibrate a method after a gate or final failure.
- Call calibration-bootstrap reuse an independent safety audit.
- Store SSH credentials in code, configuration, logs, or provenance.
- Report estimated E2E as measured E2E.

## Success Criteria

- All pure selection tests fail before implementation and pass afterward.
- Three dataset preflights reproduce source hashes and protocol identities.
- Each dataset produces exactly 10 provisional and 10 frozen selections.
- Every risk-gate comparison has 200 replicates or an exact identical-action
  shortcut.
- Final output contains exactly one frozen result per registered work point.
- Final oracle is diagnostic-only and cannot affect frozen selections.
- A cross-dataset report compares adaptive selection, global configuration,
  dataset references, and final oracle regret.
- Existing experiment hashes remain unchanged.
