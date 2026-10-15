# Implementation review

## Scope

This change adds an isolated, resumable post-hoc fusion ablation. It reads the
unified CAVS-OQR point selections and detector caches, evaluates Reversible and
MAUR on the same selections, and runs paired bootstrap. It does not modify the
source experiment, retrain a learner, or rerun detector inference.

## Correctness

- Protocol tests cover the exact method, action, role, budget, row and bootstrap matrix.
- Preflight verifies source completion hashes, every frozen selection key, allowed
  actions, capacity constraints, detector hash and role population.
- COCO MAUR uses the same `current@640` geometry and mapped ROI predictions as the
  Reversible action. Oregon uses the co-generated `current@640` MAUR cache recorded
  by the Cloud-V3 source manifest.
- Every fusion pair must have identical selection digest, action counts, request
  bytes and KB/image. Zero-ROI pairs must have exactly identical AP metrics.
- Reversible is re-evaluated and must reproduce the source point row exactly.
- Bootstrap computes MAUR minus Reversible with identical resampled images/events.

## Readability and architecture

- Pure contracts are separated in `scripts/core.py`; evaluator orchestration stays
  in `scripts/run_experiment.py`.
- Existing canonical evaluator and bootstrap implementations are reused read-only.
- No new dependency or detector-specific duplicate evaluator was introduced.

## Security and provenance

- No credentials or network endpoints are stored in the experiment.
- All externally loaded action paths come from frozen manifests and are hashed.
- Existing final roles are explicitly reported as reused post-hoc evidence.

## Performance and isolation

- The experiment uses no GPU and at most four bootstrap workers.
- Math-library threads are fixed to one by the launch command and the process runs
  at `nice=15`.
- Partial bootstrap artifacts are saved every 100 replicates and can resume.
- Oregon waits for the active unified experiment's point completion rather than
  reading an incomplete selection file.

## Verification

- Local unit tests: 5/5 passed.
- Remote unit tests in the target environment: 5/5 passed.
- COCO preflight passed and generated 209 calibration / 1,906 final MAUR actions.
- COCO full point integration passed: 24/24 rows, exact transport invariance, and
  exact Reversible source reproduction.

## Verdict

Approved for low-priority background execution. No blocking correctness,
architecture, security, or resource-isolation issue remains.
