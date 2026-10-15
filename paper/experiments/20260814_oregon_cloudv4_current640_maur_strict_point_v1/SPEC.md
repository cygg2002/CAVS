# Oregon Cloud-V4 Current@640+MAUR strict point experiment

## Objective

Repeat the frozen Oregon Cloud-V3 point protocol with Cloud-V4 epoch-29
`best.pt`, without reusing any Cloud-V3 cloud prediction or learner checkpoint.

## Exact scope

- Roles: fit 9,923 / calibration 4,961 / final 4,962.
- Actions: Edge, Original, Q90, Current@640.
- ROI fusion and training reward: MAUR.
- Learner: CurrentActionValue, 5-fold OOF plus full-fit, three seeds.
- Main output: three-seed ensemble at Fixed-byte 10%, 25%, and 40%.
- No external baselines, Random, Fixed-call, bootstrap, or method selection.

## Commands

- Test: `python -m unittest discover -s tests -v`
- Preflight: `python scripts/cache_actions.py --stage preflight`
- Run: `python scripts/run_pipeline.py`

## Boundaries

- Always generate Cloud-V4 Original/Q90/Current@640 predictions from scratch.
- Always train only on fit; calibration is evaluation-only; final is point-only.
- Never read Cloud-V3 cloud prediction caches or learner checkpoints.
- Never write into a source experiment.

## Success criteria

- Cloud model hash equals the frozen Cloud-V4 best checkpoint.
- Twelve cache jobs complete: four actions for each of three roles.
- Eighteen learner jobs complete: three seeds times five folds plus full-fit.
- Exactly three final Fixed-byte ensemble rows are produced at >=98% capacity.

