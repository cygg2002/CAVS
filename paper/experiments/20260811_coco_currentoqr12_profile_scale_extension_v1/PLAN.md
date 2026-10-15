# Implementation Plan

## Phase 1: Protocol and Tests

- [ ] Freeze canonical sources, Cloud hash, R12 action IDs, and resource gate.
- [ ] Add failing contract tests.

## Phase 2: Candidate Cache

- [ ] Build four deployable crop geometries from Edge detections.
- [ ] Reuse only byte-identical completed candidates; infer all missing scales
      after the GPU gate passes.
- [ ] Map ROI detections and apply the frozen Reversible Fusion.

## Phase 3: Reward and Learning

- [ ] Build fold-contained contextual AP50-95 rewards for O/Q/R12.
- [ ] Retrain the Reversible single-ROI and OQR12 methods with five OOF folds
      plus full-fit for three seeds.

## Phase 4: Evaluation

- [ ] Evaluate historical OQ/OQR plus both new Reversible methods at Fixed-byte
      and Fixed-call 10/25/40.
- [ ] Report AP50-95/AP50/AP75, bytes, cloud calls, and action distribution.
- [ ] Run paired bootstrap and write the Chinese report.

## Resource Isolation

- Cache inference does not begin until free GPU memory satisfies the protocol
  gate for three consecutive polls. GPU utilization may be shared after the
  user explicitly paused Cloud-V4; the 11 GB free-memory guard is unchanged.
- CPU stages run with low priority and bounded workers.
- No source experiment file is opened for writing.
