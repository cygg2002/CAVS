# Implementation plan

1. Freeze the single-image inference and payload contracts with tests.
2. Build hash-bound, per-image resumable action shards on the remote GPU.
3. Materialize COCO role bundles with MAUR as the Current@640 action output.
4. Rebuild fold-contained fit/calibration contextual rewards.
5. Train 5-fold OOF and full-fit CAVS for all three seeds.
6. Evaluate cached point results and export a non-pickle deployment policy.
7. Sync final artifacts and rerun the 5,000-image HTTP system benchmark.

Every stage writes a completion marker only after population and hash checks
pass. Restarting the pipeline resumes from completed image shards and jobs.

