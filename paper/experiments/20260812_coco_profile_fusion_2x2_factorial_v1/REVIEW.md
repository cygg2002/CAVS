# Implementation review

## Correctness

- The protocol fixes the four factorial cells, one reused anchor, three newly
  trained cells, 54 new training jobs, 72 point rows and 15 bootstrap jobs.
- Current/Historical MAUR is generated from the exact paired geometry, mapped ROI
  detections, Edge detections and payload used by Reversible.
- MAUR fit/calibration rewards are fold-contained; final rewards are forbidden.
- The Teacher gate reads calibration only. If it opens, all three missing cells are
  trained so the 2×2 remains complete instead of selecting only a favorable cell.
- Current-Reversible source checkpoints are hash-bound and its calibration/final
  metrics must reproduce the source experiment exactly.
- Bootstrap uses identical resampled images for the two Profile effects, two Fusion
  effects and factorial interaction.

## Readability and architecture

- Pure contracts are isolated in `scripts/core.py`.
- Action construction, reward generation and learner evaluation remain separate,
  resumable stages.
- Canonical evaluator, reward backend, scheduler and training adapter are reused
  read-only rather than reimplemented.

## Security and provenance

- No new dependency, credential or external write target is introduced.
- Every source is path- and hash-checked; source experiments are never assigned as
  output roots.
- Public val2017 reuse is explicitly marked post-hoc and cannot affect training,
  gating or selection.

## Performance and isolation

- The waiting orchestrator imports no model code and uses effectively zero CPU.
- It starts only after both active CAVS experiments write complete markers.
- Reward/training are capped at twelve low-priority workers; bootstrap at four; each
  numerical worker uses one math thread.
- Detector inference and GPU use are forbidden by protocol.

## Verification

- Local unit tests: 6/6 passed.
- Remote unit tests: 6/6 passed.
- Local and remote Python compilation/import checks passed.
- Read-only remote audit confirmed all Current/Historical geometry, mapped ROI and
  Reversible caches exist for fit/calibration/final with the exact Cloud hash.
- Full integration intentionally remains queued behind the active experiments to
  satisfy the user's non-interference requirement.

## Verdict

Approved for isolated queued execution. No blocking correctness, architecture,
security or resource-isolation finding remains.
