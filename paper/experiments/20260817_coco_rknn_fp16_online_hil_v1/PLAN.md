# Implementation Plan

## Phase 1: Causal Core

- [x] Implement exact rational prefix-byte accounting.
- [x] Add deterministic action-value minus price selection.
- [x] Prove suffix independence and resume equivalence with unit tests.

## Phase 2: One-Frame Runtime

- [x] Load an images-only sequence manifest without COCO ground truth.
- [x] Run one RKNN frame and construct only the enabled payloads.
- [x] Decide, immediately request the cloud when needed, fuse, and commit.
- [x] Add immutable run contracts and resumable frame logs.

## Phase 3: Methods and Controls

- [x] Load the independently trained CAVS-OQR three-seed NumPy policy.
- [ ] Load independently trained CAVS-O and CAVS-OQ three-seed policies.
- [x] Load the hash-pinned historical ORIC/MORIC pair and Confidence/Random.
- [x] Calibrate each available method/budget on calibration-only causal replay.
- [x] Refuse runs with mismatched controls, policies, seeds, or runtime assets.

## Phase 4: Training and Export

- [ ] Train 18 CAVS-O and 18 CAVS-OQ jobs from existing FP16 rewards.
- [ ] Optional: retrain three-seed RKNN-FP16 ORIC/MORIC metric variants.
- [ ] Export the pending variants as pickle-free, hash-bound policy bundles.

## Phase 5: Board Experiment

- [x] Pass local unit/integration tests and bundle verification.
- [ ] Run a fresh 50-image smoke for every method family.
- [ ] Run formal methods sequentially to avoid latency interference.
- [ ] Evaluate COCO AP on the Mac and assemble the paper tables.
