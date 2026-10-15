# Spec: COCO BudgetSet Registered Paper Evaluation v1

## Objective

Run a new, paper-level COCO evaluation of our BudgetSet ranker against
paper-faithful ORIC/MORIC reimplementations. The experiment must use images that
were not present in the completed canonical 12K selector roles or the derived
fit5000 diagnostic cache. It must rebuild contextual rewards using development
images only and access the registered final metrics once, only after an audit
gate passes.

Primary question: under the frozen YOLO26n edge detector, YOLO26x cloud
detector, Original-image upload, and exact 10%/25%/40% fixed-call budgets, does
BudgetSet outperform the strongest frozen ORIC/MORIC comparator?

Secondary question: does the conclusion remain under equal Original-byte
capacity using CostEligible scheduling?

## Frozen Data Contract

- Source population: COCO train2017.
- Exclusion population: all 12,000 train2017 image IDs in the completed
  `canonical_v1/data/coco_roles.json`. The old fit5000 action cache is a subset
  of the excluded canonical `fit` role.
- A SHA-256 ordering with seed `20260804` selects 9,500 new images:
  - development: 5,000;
  - audit: 1,500;
  - registered final: 3,000.
- Development is the only role allowed to construct contextual reward labels,
  fit models, select epochs, and construct empirical CDF targets.
- Audit is evaluation-only. It performs the go/no-go decision and freezes the
  strongest external comparator for each work point.
- Registered final is evaluated once only if audit passes. It cannot select a
  model, epoch, comparator, threshold, or success criterion.

The registered final is procedurally sealed rather than cryptographically
hidden because train2017 annotations exist on the same server. Stage gates,
completion files, immutable hashes, and code contracts enforce the boundary.

## Detector and Payload Contract

- Edge: official `yolo26n.pt`, SHA-256
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
- Cloud: official `yolo26x.pt`, SHA-256
  `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`.
- Both infer Original images at `imgsz=640`, `conf=0.001`, `iou=0.7`,
  `max_det=300`, `rect=True`, `end2end=False`.
- Upload cost is the exact source JPEG file size. No Q80/Q90 recoding and no ROI
  action is used in the primary paper comparison.
- Reporting uses COCO Official AP@[.5:.95], AP50, AP75, AP-small/medium/large,
  request KB/image, cloud-call ratio, GFLOPs, and estimated E2E@10Mbps.

## Methods

1. `ORIC-AP50`: paper-faithful top-25 feature and contextual AP50 reward.
2. `MORIC-AP50`: the same feature with the fit empirical CDF and target-weighted
   MSE.
3. `ORIC-AP5095`: metric-matched strong control using contextual AP50-95.
4. `MORIC-AP5095`: metric-matched CDF strong control.
5. `BudgetSet`: our permutation-invariant top-64 detection-set encoder,
   budget-conditioned score, and exact fixed-byte teacher-boundary objective.

The ORIC/MORIC implementation is the frozen paper-faithful reimplementation of
Qiu et al. based on official repository commit
`859f70240aa090359ba827b374b68ca7821b7d55`. It is not described as execution
of the authors' repository because detector and dataset adapters are local.

## Reward Contract

- Each target uses a deterministic context of 32 other development images.
- A single base/cloud COCO evaluation produces both AP50 and AP50-95 deltas.
- Reward is `(context_size + 1) * (cloud_metric - edge_metric)`.
- Audit and registered-final images are forbidden from every reward context.
- Reward artifacts bind the development-ID digest, detector-cache hashes,
  annotation hash, context seed, and reward implementation hash.

## Training Contract

- ORIC/MORIC retain the frozen four-layer, 16-unit MLP protocol.
- BudgetSet retains the preceding diagnostic architecture and hyperparameters:
  top-64 box set, three seeds, 80 epochs maximum, patience 10, AdamW, and the
  regression/within-image/budget-boundary objective.
- Epoch selection is development-internal only. No audit or final metric may
  alter training.
- Deployable inputs are restricted to Edge detections, image geometry, Original
  payload cost, action type, and budget. GT and cloud outputs are forbidden.

## Evaluation and Registered Gates

Work points are 10%, 25%, and 40% for both modes:

- fixed-call: exactly K images use cloud Original and all others use Edge;
- fixed-byte: total request bytes are no greater than the same fraction of the
  all-Original byte capacity.

Audit go/no-go, evaluated on fixed-call AP50-95:

- BudgetSet gains at least `+0.001` AP at two of three budgets against the best
  external method at that work point;
- no budget loses more than `0.001` AP;
- no implementation or isolation audit fails.

For each mode/budget, audit freezes the external comparator with the highest AP.
Final never reselects the comparator.

Registered-final superiority requires:

- paired image bootstrap with 1,000 deterministic replicates;
- the 95% CI lower bound of `BudgetSet - frozen comparator` is above zero at
  at least two fixed-call budgets;
- the remaining fixed-call point has CI lower bound at least `-0.001`;
- BudgetSet does not increase request KB/image by more than 2% at a fixed-call
  point;
- all fixed-byte selections respect capacity.

Point-estimate gains are reported even when the statistical gate fails.

## Commands

```bash
python -m unittest discover \
  -s paper/experiments/20260804_coco_budgetset_registered_paper_v1/tests -v

python paper/experiments/20260804_coco_budgetset_registered_paper_v1/scripts/run_paper.py \
  --stage prepare
python paper/experiments/20260804_coco_budgetset_registered_paper_v1/scripts/infer_original.py \
  --role all --detector all
python paper/experiments/20260804_coco_budgetset_registered_paper_v1/scripts/run_paper.py \
  --stage rewards
python paper/experiments/20260804_coco_budgetset_registered_paper_v1/scripts/run_paper.py \
  --stage train
python paper/experiments/20260804_coco_budgetset_registered_paper_v1/scripts/run_paper.py \
  --stage audit
python paper/experiments/20260804_coco_budgetset_registered_paper_v1/scripts/run_paper.py \
  --stage final
```

## Project Structure

```text
configs/protocol.json       frozen paths, counts, hyperparameters, and gates
scripts/paper_core.py       pure split, reward, selection, and bootstrap contracts
scripts/infer_original.py   resumable Edge/Cloud inference chunks
scripts/run_paper.py        gated prepare/rewards/train/audit/final stages
tests/                      unit and isolation tests
runs/                       new artifacts only
notes/results_zh.md         final interpretation
```

## Testing Strategy

- Unit tests for exclusion/disjointness, deterministic split, contextual reward
  isolation, deployable features, exact fixed-call and fixed-byte constraints,
  audit comparator freezing, one-time final access, and paired-bootstrap image
  remapping.
- A two-image inference smoke verifies category mapping, box coordinates,
  payload bytes, chunk resume, and both detector hashes.
- Existing ORIC/MORIC, BudgetSet, and CostEligible tests run as regressions.
- Source hashes and result hashes are checked before stage completion.

## Boundaries

- Always: write only inside this experiment; retain failed audit/final artifacts;
  report all registered budgets and both modes.
- Ask first: change detector weights, image population, reward context size,
  method hyperparameters, budgets, audit gate, bootstrap count, or success gate
  after `prepare` completion.
- Never: use audit/final labels for training; inspect final aggregate metrics
  before audit passes; run final twice; select the final comparator post hoc;
  call this a hidden COCO test-dev result.

## Success Criteria

- Exactly 9,500 new IDs with zero overlap against the old 12K.
- Complete Edge and Cloud prediction caches for all roles with frozen hashes.
- Contextual rewards contain only development IDs and contexts.
- All five methods produce six registered point estimates on audit and final.
- Audit and final gates execute exactly as specified.
- Final report contains paired-bootstrap intervals and a clear pass/fail claim.

## Open Questions

None. The protocol is frozen before metrics are produced.
