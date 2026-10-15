# CAVS final paper-completion experiment

## Objective

Close the remaining COCO and Oregon algorithmic/statistical gaps without changing
the frozen detector, split, action cache, evaluator, or already running experiments.
The experiment is a post-hoc paper-completeness extension; neither public COCO
val2017 nor the already opened Oregon final is represented as a new hidden test.

## Frozen method names

- `CAVS-OQ-Core`: independently trained CurrentActionValue router on
  Edge/Original/Q90. This is the paper core method.
- `CAVS-OQ-from-OQR`: an OQ mask applied to a separately trained OQR router. It is
  an ROI attribution control and must never be renamed to `CAVS-OQ-Core`.
- `CAVS-OQR-MAUR`: Current@640 ROI with MAUR predictions, trained against MAUR
  contextual AP50-95 rewards. This is the optional ROI extension.
- `Raw-ActionValue-OQ`: historical diagnostic only. It is not an alias of the core.

## Datasets and immutable sources

### COCO

- split: canonical train2017 fit=9,678; calibration=571; val2017=5,000.
- cloud model SHA-256:
  `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`.
- three model seeds: 20260809, 20260810, 20260811.
- Current@640+MAUR checkpoints and rewards are reused from the completed strict
  2x2 controlled retraining only after hashes and protocol equality pass.

### Oregon

- same-camera event-disjoint fit=9,923; calibration=4,961; final=4,962.
- Cloud-V3 epoch-28 SHA-256:
  `d3890a57fc7d567880a5b01b7eb949387d9b7058ce7ee25bdb9471d9bb4642d9`.
- three model seeds: 20260810, 20260811, 20260812.
- Current@640+MAUR is newly trained because the existing same-selection MAUR
  experiment did not train against MAUR rewards.

Identity, not a bare numeric image id, is the split-leakage key. ENA/Oregon-style
numeric image-id reuse across roles must not be interpreted as overlap.

## Required comparisons

1. `CAVS-OQ-Core` versus Confidence, Random-50, ORIC-AP5095, and MORIC-AP5095.
2. Three single models, mean +/- sample SD, and three-model ensemble for CAVS,
   ORIC and MORIC.
3. Same-backbone `TwoStage-OQ`: use the exact CAVS action-value scores, first
   choose each image's best cloud action, then rank/select images. No new network.
4. Same-backbone `CostInValue-OQ`: retrain the same CurrentActionValue model while
   exposing payload fields to the value network. The scheduler is unchanged.
5. `CAVS-OQR-MAUR` versus `CAVS-OQ-from-OQR`, Reversible OQR, external methods,
   Confidence, and Random-50.
6. Existing O/OQ/OQR and Profile x Fusion results are imported by hash, never
   recomputed or silently relabeled.

## Evaluation

- primary: Fixed-byte 10%, 25%, 40%; capacity utilization must be >=98% unless
  infeasibility is proven.
- secondary: Fixed-call 10%, 25%, 40%.
- metrics: AP50-95, AP50, AP75, KB/image, action counts and capacity utilization.
- COCO bootstrap unit: image.
- Oregon bootstrap units: event and image.
- 1,000 paired bootstrap replicates per comparison.
- Random uses 50 complete frozen selections. In every image/event bootstrap
  replicate, Random AP is evaluated for all 50 selections and averaged before the
  CAVS-minus-Random delta is formed. Averaging selections is forbidden.

## Resource and mutation boundary

- Completed source experiments are read-only.
- Existing running jobs are not stopped, reniced, rebound, or edited.
- Every worker uses one Torch/OMP/MKL thread.
- New computation starts at nice=19 with at most four workers while protected
  tasks are active. The launcher can increase to the configured maximum only after
  their completion markers appear.
- Final images are used only for frozen prediction/evaluation/bootstrap, never for
  model fitting, calibration, threshold or method selection.

## Success criteria

- exact source hashes and role counts pass preflight;
- all declared training jobs have atomic checkpoints, scores and completion files;
- Fixed-byte capacity is never exceeded;
- score/image identity alignment is one-to-one within each dataset/role;
- no final role enters reward, fold, training, model-selection or calibration code;
- all point rows and bootstrap comparison ids are unique and complete;
- rerunning any completed stage is idempotent.

## Out of scope

RK3588 latency/energy/network measurements and a genuinely new hidden test require
external hardware/data authority and are not fabricated by this CPU cache run.

