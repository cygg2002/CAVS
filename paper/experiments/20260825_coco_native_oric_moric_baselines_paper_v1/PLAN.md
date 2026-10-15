# Implementation Plan

## Phase 1: Contracts

- [ ] Freeze source hashes, role sizes, method labels and job matrices.
- [ ] Test Original-only action maps, point counts and bootstrap counts.

## Phase 2: Evaluation

- [ ] Import hash-verified scalar scores and frozen CAVS-OQR selections.
- [ ] Evaluate native ORIC/MORIC/Confidence/Random at both budget modes.
- [ ] Report ORIC/MORIC individual seeds, mean/SD and ensembles.

## Phase 3: Statistics and Report

- [ ] Run CAVS-OQR against each native baseline at three Fixed-byte budgets.
- [ ] Use nested Random-50 bootstrap rather than a single random seed.
- [ ] Produce a Chinese report separating native and OQR comparisons.

## Checkpoint

- [ ] Tests and remote preflight pass.
- [ ] Point and bootstrap completion markers are present.
- [ ] No source experiment is modified and no RKNN path is read.

## Risks

| Risk | Mitigation |
|---|---|
| Native and OQR action spaces are conflated | Separate table headings and explicit action-space columns. |
| Old ORIC/MORIC scores are accidentally used | Pin current 20260813 three-seed score hashes. |
| Random uncertainty is understated | Recompute all 50 Random APs inside every bootstrap sample. |
