# Specification: Three-Dataset DCSB and ORIC/MORIC External Baselines

## Objective

Implement isolated, paper-level external baselines for COCO, ENA24, and Oregon
without modifying any completed experiment. The package reads the frozen roles,
detector predictions, payload bytes, SharedAction model, and Official AP evaluator
from the completed 20260720 protocol and writes only below this experiment's
`runs/` directory.

The external methods are:

1. `DCSB-Native`: the source-paper three-threshold binary rule, reported at its
   natural upload rate.
2. `DCSB-Score-Adapted`: a continuous margin derived from the same DCSB rule so
   it can be evaluated at the paper's ten registered work points. This is an
   explicit scheduling adaptation, not the original DCSB algorithm.
3. `ORIC-MLP`: top-25 weak-detector output features and an MLP regressor trained
   on contextual mAP rewards.
4. `MORIC-MLP`: the same features and MLP architecture, with ORIC rewards mapped
   through the fit empirical CDF and a high-reward weighted MSE objective.

`ORIC-MLP` and `MORIC-MLP` are paper-faithful reimplementations based on the
paper and its official repository. They are not claimed to reproduce the
authors' published numerical results because detectors, datasets, splits, and
the final evaluator differ.

## Evidence Sources

- DCSB: Cao et al., ICDCS 2023, DOI `10.1109/ICDCS57875.2023.00062`, open
  preprint `arXiv:2108.12858`.
- ORIC/MORIC: Qiu et al., SEC 2024, DOI
  `10.1109/SEC62691.2024.00021`, open preprint `arXiv:2410.18919`.
- Official ORIC/MORIC code:
  `https://github.com/qiujiaming315/edgeml-object-detection`, inspected commit
  `859f70240aa090359ba827b374b68ca7821b7d55` on 2026-07-28.

The implementation choices and deviations are frozen in
`configs/literature_sources.json`.

## Frozen Data Contract

- Datasets: COCO, ENA24, Oregon.
- Roles: `fit`, `calibration`, `final`.
- Fit is the only role used for DCSB parameter fitting, ORIC reward generation,
  empirical-CDF construction, and MLP fitting.
- Calibration does not train DCSB or ORIC/MORIC. It is used for Online
  thresholds and for the existing Fixed-byte lambda objective, including the
  frozen calibration teacher gain required by that scheduler.
- Final is evaluation-only and is never used to choose a method, fit a model,
  tune a threshold, or repair a result.
- Expected role sizes:
  - COCO: fit 9,678; calibration 571; final 5,000.
  - ENA24: fit 1,400; calibration 400; final 1,758.
  - Oregon: fit 9,923; calibration 4,961; final 4,962.

All files actually read are hashed before and after a stage. A hash change
aborts completion.

## DCSB Contract

The source-faithful rule uses weak-detector outputs only at inference:

1. Fit a confidence threshold whose total weak-detector box count most closely
   matches the total fit ground-truth count.
2. Let `estimated_count` use that threshold and `detected_count` use 0.5.
3. If the counts are equal, classify the image as easy.
4. Otherwise classify it as difficult when `estimated_count` exceeds the fitted
   count threshold or the minimum estimated box-area ratio is below the fitted
   area threshold.
5. Fit count and area thresholds to predict whether the Original cloud detector
   finds more confidence-0.5 boxes than the edge detector on fit.

The source code's potentially non-terminating binary search is replaced by an
exact deterministic search over observed confidence breakpoints. This preserves
the stated count-matching objective.

`DCSB-Score-Adapted` ranks uncertain images by the maximum normalized violation
of the count and area thresholds; images for which estimated and detected counts
are equal remain below all uncertain images. This continuous score exists only
to support the common budget protocol.

## ORIC/MORIC Contract

- Input: class histogram plus the normalized `center-x, center-y, w, h,
  confidence` of the top 25 edge detections, ordered by confidence.
- ORIC reward for target image `i`:

  `(|E| + 1) * (mAP(E_edge + i_cloud) - mAP(E_edge + i_edge))`.

- `E` is a deterministic fit-only context excluding `i`.
- Primary contextual reward uses AP@0.5, matching the public source code's
  default. Final reporting still uses the frozen Official AP@[.5:.95].
- MORIC target is the right-continuous empirical CDF of fit ORIC rewards.
- ORIC uses ordinary MSE. MORIC uses target-weighted MSE.
- MLP: four 16-unit hidden layers, ReLU, 0.1 dropout, Adam, 100 epochs,
  learning rate 0.005, weight decay 0.00005, and the source milestones.
- A deterministic fit-internal holdout selects the epoch; the selected epoch is
  then retrained using all fit samples. No calibration/final label is used.

ORIC reward generation is checkpointed and resumable because context evaluation
is the dominant cost.

## Payload and Budget Contract

Each score method is evaluated with:

- `Original`: source-faithful whole-image upload.
- `SharedAction`: controlled diagnostic using the already-frozen common action
  head. This isolates ranking quality but is not the original external method.

Registered work points:

- Fixed-call: 10%, 25%, 40%.
- Online: nominal 10%, 25%, 40%, with thresholds frozen on calibration.
- Fixed-byte: 5%, 10%, 20%, 40% of all-Original byte capacity, with lambda
  frozen on calibration.

The final external matrix is 3 continuous score methods x 2 payload policies x
10 work points = 60 rows per dataset, plus two native DCSB rows and three system
references, for 65 rows per dataset.

## Commands

```bash
python scripts/run_dataset.py --dataset coco --stage preflight
python scripts/run_dataset.py --dataset coco --stage rewards
python scripts/run_dataset.py --dataset coco --stage evaluate
```

```bash
python scripts/run_pipeline.py --datasets coco ena24 oregon --stage all
python3 -m unittest discover -s tests -v
```

## Boundaries

### Always

- Write only inside this experiment directory.
- Preserve source file hashes and record runtime versions.
- Keep `DCSB-Native` separate from `DCSB-Score-Adapted`.
- Label `Original` as the external-method payload and `SharedAction` as a
  controlled diagnostic.
- Preserve failures and incomplete reward checkpoints.

### Never

- Modify or regenerate any completed 20260720-20260727 result.
- Train on calibration, audit, or final labels.
- Call the existing internal `MORIC-style` Ranker a faithful MORIC baseline.
- Claim official reproduction or direct superiority under the original paper's
  detector/data protocol.
- Select a model or repair a threshold after viewing final results.

## Success Criteria

- Unit tests cover feature extraction, DCSB rules, contextual reward, empirical
  CDF ties, scheduling, protocol counts, and model-training/final isolation.
- Preflight validates all expected roles and frozen input paths.
- Every dataset produces exactly 65 unique aggregate rows.
- All ten work points exist for each continuous method/payload pair.
- DCSB native upload rate is reported without forcing it to a target budget.
- Source hashes are identical before and after every completed stage.
- Reward completion binds the protocol hash, fit-source digest, reward artifact,
  and fitted DCSB model; evaluation rejects any stale or replaced bundle.
- A dataset is marked complete only after rewards and evaluation both pass.
