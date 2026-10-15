# COCO RKNN-FP16 final policy completion v1

## Objective

Produce the two policy families missing from the final causal-online RK3588
evaluation without modifying or reusing an incompatible learner checkpoint:

1. an independently trained `CAVS-OQ` policy over
   `Edge / Original / Q90`; and
2. RKNN-domain, three-seed `ORIC` and `MORIC` policies over
   `Edge / Original`.

The outputs are inputs to
`20260828_coco_rknn_fp16_online_hil_final_v2`; this experiment does not run or
select on the final hardware results.

## Frozen data and models

- Dataset roles: fit = 9,678, calibration = 571, final = 5,000.
- Edge detector: `yolo26n-rk3588-fp16.rknn`, SHA-256
  `0915a99235c3be5723fcfeda45b06775ecccc2908cc5b490f093f23bbdede12d`.
- Cloud detector: `yolo26x.pt`, SHA-256
  `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`.
- Source candidates and rewards are hash-pinned from
  `20260816_coco_rknn_cavs_oqr_deployment_retrain_v1`.
- Seeds: `20260809`, `20260810`, `20260811`.
- Frozen five folds from the existing canonical COCO split.
- Calibration labels never train a policy. Final labels and final scores never
  select an epoch, hyperparameter, controller, or method.

## CAVS-OQ protocol

- Train the same ActionValue architecture and hyperparameters as CAVS-OQR.
- Filter the source action records to exactly `Edge / Original / Q90` before
  forming the reward matrix or model inputs.
- Run five-fold OOF plus one full-fit model for each of three seeds: 18 fresh
  jobs in total.
- Do not initialize from, mask, or otherwise reuse CAVS-OQR weights.
- Report calibration and final point results for each single seed and the
  three-seed score ensemble at 10%, 25%, and 40% fixed-byte budgets.

## ORIC/MORIC protocol

- Use the public ORIC output feature: 80 class counts plus the top-25
  `(cx, cy, w, h, confidence)` rows, dimension 205.
- Recompute the ORIC target from the RKNN-FP16 Edge detections and frozen
  Original cloud detections. Use AP50, 1,000 fit-only context images, and fixed
  context seed `20260728`, matching the faithful baseline implementation.
- ORIC regresses the raw reward with unit sample weights.
- MORIC regresses the right-continuous empirical CDF rank of that reward and
  uses the rank as the sample weight.
- Train three full-fit MLPs per method with identical architecture and
  hyperparameters, changing only the preregistered seed.
- Report single-seed mean/SD and the three-seed score ensemble. Only the
  ensemble is used for the main board run; single seeds are retained for the
  fairness audit.

## Export and acceptance

- Export pickle-free, hash-pinned NPZ policies.
- Verify PyTorch versus NumPy score parity with maximum absolute error
  `<= 1e-5` for every seed and method.
- Manifests must bind the role populations, model hashes, source hashes,
  action space, seeds, feature contract, reward contract, and training status.
- All unit tests, preflight checks, 18 CAVS-OQ jobs, six external MLP jobs,
  point-result rows, and export parity checks must complete before an artifact
  can be copied into the final board bundle.
- Existing CAVS-OQR, historical ORIC/MORIC, board runs, and source experiment
  files are read-only.

