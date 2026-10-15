# Specification: Three-Dataset Nine-Ranker Hierarchical-TriAction Evaluation

## Objective

Create one isolated exploratory paper-level evaluation for COCO, ENA24, and
Oregon with a common Ranker/action separation, without modifying or replacing
any existing canonical or exploratory experiment.

The requested action policies are:

1. `SharedAction`
2. `CostEligible-AllROI`
3. `GBDT-Hierarchical`

The requested deployment modes and budgets remain identical to the previous
paper-level protocol:

- Fixed-call: 5%, 10%, 15%, 25%, 40%, 50%, 65%, 80%.
- Online: nominal 10%, 25%, 40%.
- Fixed-byte: 5%, 10%, 20%, 40% of Original-byte capacity.
- Primary safety budgets: Fixed-call 10%, 25%, 40%.
- Official AP tolerance: every constrained calibration point must satisfy
  `Delta AP >= -0.001`.

## Compatibility Gate

The eight current canonical Rankers share the current dataset detector, raw
FPN, current fit-only PCA, features, actions, and evaluator.

The historical `Legacy-Frozen-32D` models are not equally compatible:

| Dataset | Historical Legacy input | Current canonical input | Verdict |
|---|---|---|---|
| ENA24 | 448D, Edge SHA `94233054...20f53` | same 448D and same SHA | Exact-compatible |
| COCO | 896D YOLO26s, SHA `646f8bc3...84a1b` | 448D YOLO26n, SHA `9b09cc8b...d4fef` | Shape/checkpoint incompatible |
| Oregon | 448D location-disjoint Edge, SHA `252250e8...082a` | 448D event-disjoint Edge, SHA `b975f9f8...a2c4b` | Shape-compatible but checkpoint/split incompatible |

Therefore the literal nine-Ranker matrix requested by name is not an
apples-to-apples current-protocol structure comparison on COCO/Oregon.

### Approved strict primary route

Use `Current-Controlled-32D SemanticRank` as the ninth cross-dataset primary
Ranker. It must be fitted independently for every dataset using:

- the same current canonical raw FPN and sample identities as the other current
  Rankers;
- a fit-only 32D semantic representation;
- current fit targets and no final/audit selection;
- the same current SharedAction and evaluator.

Keep `Legacy-Frozen-32D` outside the primary matrix as a historical bridge:

- ENA24: exact-compatible paper-level bridge result.
- COCO: historical YOLO26s bridge only if executed, explicitly charging its additional
  feature-extractor cost and excluding it from shared-threshold constraints.
- Oregon: out-of-distribution frozen transfer only if executed, excluding
  it from shared-threshold constraints and never calling it a controlled
  structure comparison.

### Rejected primary route

Running all three historical models frozen is permitted only as an
explicit transfer/bridge diagnostic:

- COCO uses historical YOLO26s 896D features and is not the current edge
  architecture.
- Oregon applies an old location-disjoint PCA/ranker to current event-disjoint
  FPN activations from a different checkpoint.
- Neither may support a “32D vs 49D under identical conditions” claim.

This route must not be silently presented as the strict primary matrix.

## Primary Architecture

```text
Dataset adapter
  -> canonical fit/calibration/audit/final roles
  -> current detector predictions, payloads, ROI profile, evaluator
  -> current Ranker scores + approved ninth Ranker score

Ranker score
  -> Fixed-call / Online / Fixed-byte scheduler

Frozen SharedAction
  -> the exact canonical action map, including any canonical ROI already used

CostEligible-AllROI
  -> overlay ROI only when ROI is available and costs less than the actual
     canonical base action

GBDT-Hierarchical
  -> one fit-only SafeROI GBDT per dataset
  -> one mode-specific threshold per dataset shared across all constrained
     Rankers
```

The algorithm is shared across datasets. Numerical model parameters and
thresholds are dataset-specific because score distributions are not
cross-dataset comparable.

The SafeROI model input is frozen at 54D in this exact order: current 49D,
Q90/Original byte ratio, ROI/Original byte ratio, ROI area ratio, Q90
availability, and ROI availability. Labels use `epsilon=0.02` against the
actual frozen SharedAction base. HistGradientBoosting parameters are frozen in
`configs/protocol.json` and checked against the source implementation hash.

## Calibration Contract

### Fixed-call

- Random is excluded from constraints.
- A single dataset/mode threshold is constrained across every non-random
  primary Ranker at 10%, 25%, and 40%.
- Require every point to satisfy `Delta AP >= -0.001` versus SharedAction.
- Select minimum mean request KB, then maximum mean Official AP.
- Apply the frozen threshold unchanged to all eight final budgets.

### Online

- Freeze one frame threshold per Ranker/target on calibration.
- Select one dataset-level ROI threshold across all non-random Rankers and all
  three targets using the Fixed-call safety/cost objective.
- Preserve final upload-rate drift unchanged.

### Fixed-byte

- Recalibrate lambda for every Ranker, action policy, ROI-threshold candidate,
  and byte budget on calibration.
- Each lambda keeps the canonical objective: maximum teacher gain, then fewer
  bytes, then lower lambda. This is separate from ROI-threshold selection.
- Require all calibration AP deltas versus separately calibrated SharedAction
  to be at least `-0.001`.
- Select maximum mean Official AP, then lower mean bytes.
- Freeze threshold and lambdas before final.

### Per-Ranker ablation

Report independently selected ROI thresholds per Ranker and mode. This is an
ablation only and cannot replace the shared-threshold primary result.

## Evaluation Matrix

For a strict nine-Ranker primary matrix:

- 8 constrained non-random Rankers x 3 policies x 15 work points = 360 rows
  per dataset.
- Random x 3 policies x 8 Fixed-call budgets = 24 rows per dataset.
- Primary total = 384 rows per dataset, 1,152 across three datasets.
- Per-Ranker GBDT ablation = 120 rows per dataset, 360 total.
- Random raw seeds = 1,200 rows per dataset.
- Primary paired bootstrap = 8 constrained Rankers x 3 budgets = 24
  comparisons per dataset; 72 comparisons total, each with 200 replicates.

## Data and Statistical Boundaries

- COCO canonical roles: fit 9,678; calibration 571; audit 630; final val2017
  5,000.
- ENA24 canonical roles: fit 1,400; calibration 400; final 1,758.
- Oregon canonical roles: fit 9,923; calibration 4,961; audit 9,923; final
  4,962.
- Audit never participates in selection or fallback.
- Final never participates in fitting, threshold selection, lambda selection,
  fallback, or method choice.
- COCO val2017 and ENA24 final are fixed diagnostic benchmarks, not untouched
  hidden tests.
- Oregon registered final was already accessed by the canonical experiment.
  This new result is post-hoc exploratory final reuse, not a second registered
  final.
- Bootstrap remains image-level because the existing paired evaluator does
  not implement cluster resampling; the limitation must be reported for
  Oregon.
- All 24 comparisons use the same deterministic bootstrap seed within a
  dataset and are labeled `GBDT-Hierarchical - SharedAction`.
- E2E and selector parameter tables keep the completed 20260722 TriAction
  convention: Controlled-32D is charged 0.08 ms and 32 rank weights, while
  additional GBDT action-head inference overhead is reported separately and
  is not added to E2E.

## Commands

Planned interface:

```bash
python scripts/run_dataset.py --dataset coco --stage preflight
python scripts/run_dataset.py --dataset ena24 --stage preflight
python scripts/run_dataset.py --dataset oregon --stage preflight
```

```bash
python scripts/run_dataset.py --dataset DATASET --stage core
python scripts/run_dataset.py --dataset DATASET --stage bootstrap
```

```bash
python3 -m unittest discover -s tests -v
```

## Project Structure

```text
configs/protocol.json       Frozen shared method and budget contract
scripts/evaluation_core.py  Pure policy/calibration/matrix helpers
scripts/run_dataset.py      One isolated/resumable dataset runner
tests/                      Unit and integration contracts
runs/{dataset}/             Dataset-isolated machine-readable outputs
notes/                      Compatibility audit and Chinese results
```

## Code Style

- Python 3.10 compatible.
- Reuse the existing ENA24 exploration helpers; do not copy the entire
  1,000-line runner.
- Explicit image-ID alignment and stable ordering.
- No new dependency.
- One dataset process owns only its own output directory.

## Testing Strategy

Tests must be written before implementation and cover:

1. protocol matrix size and Random exclusion from calibration constraints;
2. exact `+Infinity -> SharedAction` and
   `-Infinity -> CostEligible-AllROI` identities;
3. selected image IDs unchanged across action policies for Fixed-call/Online;
4. Fixed-byte lambda is recalibrated per policy/candidate;
5. dataset/model/FPN compatibility checks fail on mismatched shape or detector
   hash rather than silently transferring;
6. current eight-Ranker SharedAction rows reproduce canonical results within
   `1e-12`;
7. ENA24 Current-Controlled-32D SharedAction reproduces its completed control
   experiment at 10%/25%/40% within `1e-12`;
8. frozen input hashes remain identical before and after each run.

## Boundaries

### Always

- Write to this new experiment only.
- Hash every file actually read.
- Freeze all choices before final.
- Preserve all failures unchanged.
- Record compatibility class, source model hashes, feature dimensions,
  thresholds, lambdas, selections, runtime versions, and completion markers.

### Ask first

- Running a historical extra-detector transfer as a primary result.
- Retraining detectors or changing any canonical split/action cache.

### Never

- Modify existing canonical or completed exploration directories.
- Truncate/pad COCO 896D features to 448D.
- Apply Oregon old PCA to a new checkpoint without labeling it transfer.
- Use audit/final to select or repair a failed threshold.
- Present post-hoc Oregon results as a new registered final.

## Success Criteria

1. The approved compatibility route is frozen as Current-Controlled-32D in the
   primary matrix and Legacy only in a bridge appendix.
2. Three preflights verify sample IDs, dimensions, model hashes, ROI profiles,
   and source completeness.
3. All current SharedAction rows reproduce canonical outputs within `1e-12`.
4. Every final matrix has the exact expected unique-key count.
5. Every primary calibration candidate obeys the all-Ranker AP constraint.
6. Random uses exactly 50 deterministic seeds.
7. Bootstrap contains the exact unique comparison set and 200 replicates each.
8. Input hashes match before/after and existing experiment hashes do not
   change.
9. `COMPLETED` is written only after all checks pass.

## Approved Decision

The user approved `Current-Controlled-32D SemanticRank` as the ninth primary
Ranker on 2026-07-23. Historical `Legacy-Frozen-32D` is excluded from every
primary threshold constraint and claim; ENA24 exact reproduction and the two
incompatibility records form the bridge appendix.
