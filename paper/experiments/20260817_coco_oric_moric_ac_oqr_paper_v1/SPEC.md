# Spec: COCO ORIC/MORIC Action-Conditioned OQR Paper Experiment

## Objective

Train and evaluate paper-domain action-conditioned adaptations of ORIC and
MORIC on the frozen COCO protocol. This experiment answers whether the ORIC
Top-25 representation and MORIC reward transform remain competitive when they
are given the same Original/Q90/Current@640 action space and exact-capacity
scheduler as CAVS.

The new methods are named `ORIC-AC-OQR` and `MORIC-AC-OQR`. They are adaptations
implemented in this project, not official ORIC/MORIC methods.

## Protocol

- Dataset: COCO `fit=9,678 / calibration=571 / final=5,000`.
- Domain: frozen PyTorch paper caches only. RKNN/RK3588 candidates, rewards,
  checkpoints and deployment results are forbidden.
- Actions: Edge, Original, Q90 and Current@640 with MAUR fusion.
- Reward: the same contextual AP50-95 matrix used by `CAVS-Current-MAUR`.
- Training: three seeds, five-fold OOF and one full-fit job per seed.
- Scheduling: the same Fixed-byte/Fixed-call exact-capacity scheduler at
  10%/25%/40%.
- Statistics: single-model results, mean/SD, three-model ensemble and 1,000
  paired image-bootstrap replicates.

## Method Contract

Both new methods use the ORIC paper's 205-dimensional input: 80 class counts
plus Top-25 boxes with normalized center, size and confidence. They share the
same cost-free action encoder, joint head, optimizer, inner split and
action-value loss. The only target difference is:

- `ORIC-AC-OQR`: raw contextual AP50-95 action reward.
- `MORIC-AC-OQR`: one fold-local empirical CDF fitted jointly over every valid
  cloud-action reward in the training partition.

The CDF is never fitted separately by action and never observes an OOF,
calibration or final target. Payload bytes are excluded from the value network
and are used only by the external scheduler.

## Comparisons

The final table contains:

- `CAVS-OQR`: frozen `CAVS-Current-MAUR` paper result.
- `ORIC-AC-OQR` and `MORIC-AC-OQR`: newly retrained methods.
- `ORIC-SharedAction-OQR-MAUR` and `MORIC-SharedAction-OQR-MAUR`: frozen
  three-seed scalar-ranker controls from the final-completion experiment.

Paired bootstrap covers CAVS versus each action-conditioned method, each
action-conditioned method versus its scalar SharedAction control, and ORIC-AC
versus MORIC-AC.

## Commands

- Test: `python -m unittest discover -s tests -v`
- Preflight: `python scripts/run_experiment.py --stage preflight`
- Train: `python scripts/run_experiment.py --stage train`
- Full pipeline: `python scripts/run_pipeline.py --start-now`

## Project Structure

- `configs/protocol.json`: frozen populations, source hashes and resources.
- `scripts/core.py`: protocol, CDF and job-matrix contracts.
- `scripts/methods.py`: action-conditioned ORIC/MORIC model and trainer.
- `scripts/run_experiment.py`: preflight, training, evaluation and bootstrap.
- `scripts/run_pipeline.py`: resumable low-priority remote launcher.
- `tests/`: pure contracts plus a small synthetic training integration test.
- `runs/`: experiment-local outputs only.

## Boundaries

- Always: verify source hashes and the three CAVS point values before training.
- Always: preserve final-data isolation and save fold/seed-specific artifacts.
- Always: use `nice=19`, one math thread per worker and at most eight workers.
- Never: run detector inference, regenerate rewards, tune on final, or import
  any `rk3588`, `rknn`, `online_hil` or deployment-retrain artifact.
- Never: modify or stop existing experiments on the remote server.

## Success Criteria

1. Preflight proves the paper-domain source hashes, role counts, action space,
   CAVS points and forbidden-domain checks.
2. Exactly 36 independent jobs finish: two methods x three seeds x
   (five folds + full fit).
3. OOF, calibration and final ensemble results plus final single-seed results
   are reproducible from saved score files.
4. All 15 paired-bootstrap jobs finish with 1,000 image replicates each.
5. No source experiment is modified and no RKNN/RK3588 artifact is read.

## Open Questions

None. The user explicitly selected the paper PyTorch protocol and remote CPU
training rather than the RKNN deployment domain.
