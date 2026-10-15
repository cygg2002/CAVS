# Spec: ENA24/Oregon DualHead cross-dataset evaluation

## Objective

Run two cache-only CPU experiments on the canonical ENA24 and Oregon protocols.

1. **Track A — learner generalization diagnostic.** Compare Current-BudgetSet,
   SetValue, DualHead, and ActionValue-OQ-Joint using fit-only training,
   5-fold out-of-fold predictions, and one frozen calibration evaluation.
2. **Track B — strict summary-table extension.** Add one new ranker,
   DualHead-Value, to the frozen SharedAction, CostEligible-AllROI, and
   GBDT-Hierarchical policies used by the existing ENA24/Oregon result tables.

No detector inference is allowed. Existing canonical action caches and frozen
action policies are read-only inputs.

## Assumptions

- ENA24 uses `canonical_v1`: fit=1,400, calibration=400, final=1,758.
- Oregon uses `canonical_oregon_event_v1`: fit=9,923,
  calibration=4,961, final=4,962.
- Oregon Cloud-V2 is a different protocol and is forbidden here.
- DualHead means the existing top-64 permutation-invariant detection-set model,
  not a newly tuned architecture.
- Track B uses only the DualHead value score to rank images. Its cost head cannot
  alter the three frozen action policies.
- Existing final sets have already been accessed. Track B is a post-hoc table
  extension, not a new hidden final. Track A never reads final.

## Tech stack

- Python 3.10/3.12, NumPy, PyTorch, scikit-learn, faster-coco-eval.
- Reused implementations from the completed canonical, DualHead, TriAction,
  and contextual-reward experiments.
- CPU-only execution; no new dependencies and no detector/GPU inference.

## Commands

```bash
python -m unittest discover -s tests -v
python scripts/run_crossdataset.py --stage preflight
python scripts/run_crossdataset.py --stage all --workers 24
```

## Project structure

```text
configs/protocol.json       Frozen protocol and source paths
scripts/crossdataset_core.py Pure contracts and matrix validation
scripts/run_crossdataset.py Resumable stage runner/orchestrator
tests/                      Unit and contract tests
runs/                       Per-dataset checkpoints, scores, results, logs
notes/                      Generated Chinese result summary
```

## Experiment design

### Shared reward and representation

- Build top-64 Edge detection sets with the existing encoder and category map.
- Compute deterministic contextual AP50-95 proxy rewards from cached detections
  with a 32-image, fit-only Edge context.
- Track A candidates are Edge, Original, and Q90. ROI is intentionally excluded
  from ActionValue-OQ-Joint so the experiment tests image/action learning without
  conflating the known dataset-specific ROI policies.
- Train seeds are 20260804, 20260805, and 20260806.

### Track A

- Roles: fit OOF and calibration only.
- Models: Current-BudgetSet, SetValue, DualHead, ActionValue-OQ-Joint.
- 5 deterministic folds; all contextual targets for a target image are computed
  from the frozen fit-only context and never from calibration/final.
- Primary: Fixed-byte 10/25/40%; secondary: Fixed-call 10/25/40%.
- Current-BudgetSet, SetValue, and DualHead rank images and use frozen
  SharedAction. ActionValue-OQ-Joint jointly selects Original/Q90.
- Report AP50-95, AP50, AP75, KB/image, action ratios, capacity use, seed
  agreement, contextual-reward correlation, and Teacher regret.

### Track B

- Role: the already-used canonical final role.
- Ranker: the frozen three-seed DualHead value ensemble trained on fit only.
- Fixed-call 10/25/40%, matching the two result-summary documents.
- Policies and fixed thresholds:
  - SharedAction: frozen canonical action head/gate.
  - CostEligible-AllROI: frozen cheaper-than-base eligibility with all eligible
    ROI actions enabled.
  - GBDT-Hierarchical: reuse the completed GBDT checkpoint and fixed-call
    threshold (`0.9028747070995273` ENA24,
    `0.9724118058752602` Oregon).
- Selected image membership must be identical across the three policies for a
  dataset/budget; only the selected action may differ.
- Compare with the strongest existing ranker in the same dataset/policy/budget
  using 200 paired image bootstraps, matching the source protocol.

## Testing strategy

- Unit tests for protocol constants, folds, score alignment, action filtering,
  capacity safety, and expected row matrices.
- Integration preflight against all canonical manifests, action caches, frozen
  GBDT models, thresholds, and source hashes.
- Smoke run on a tiny deterministic subset before the full background launch.
- Atomic/checkpointed stage outputs; rerunning resumes completed stages.

## Boundaries

### Always

- Align scores explicitly to each role's sample order and validate the exact ID
  population. ENA24 IDs may repeat across roles, so role identity is part of the
  provenance.
- Freeze inputs by SHA-256 in preflight and verify them before later stages.
- Keep action-policy checkpoints and thresholds unchanged.
- Record capacity use and assert that Fixed-byte never exceeds capacity.

### Never

- Modify any completed source experiment.
- Read final during rewards, OOF, training, calibration, or policy selection.
- Refit/reselect SharedAction, CostEligible-AllROI, or GBDT-Hierarchical.
- Import Oregon Cloud-V2 caches/models/actions.
- Describe Track B as a fresh registered or hidden final.

## Success criteria

- Tests and both dataset preflights pass.
- Track A produces exactly 96 unique aggregate rows:
  2 datasets × 2 roles × 4 methods × 2 modes × 3 budgets.
- Track B produces exactly 18 unique primary rows:
  2 datasets × 3 policies × 3 budgets.
- All Fixed-byte rows stay within capacity and all Track B Fixed-call counts
  equal the canonical rounded counts.
- Track B policy membership is identical within each dataset/budget.
- No Track A source/provenance includes a final path.
- Oregon outputs are labeled `posthoc_exploratory_final_reuse`; all Track B
  outputs are labeled post-hoc.

## Interpretation gate

This experiment is primarily comparative, not a new-paper gate. A cross-dataset
main-method candidate is supported only if it is non-inferior (within 0.001 AP)
to the strongest internal candidate at all three Fixed-byte budgets on both
calibration sets and improves at least two dataset-budget points by 0.001 AP.
Track B is reported descriptively against the frozen historical tables and is
not used to tune the learner.
