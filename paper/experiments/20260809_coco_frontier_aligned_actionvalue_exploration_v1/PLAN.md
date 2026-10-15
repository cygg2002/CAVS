# Plan: COCO Frontier-Aligned ActionValue Exploration v1

## Phase 1: Evidence before implementation

- [x] Audit current OOF reward, score, payload, and selection errors.
  - Acceptance: report membership regret, action regret, boundary-band error,
    oracle ceilings, and per-budget shadow prices for OQ and OQR.
  - Verify: exact 9,678-image coverage and no forbidden role reads.
- [x] Search adjacent literature and record an evidence-bounded novelty map.
  - Acceptance: cover decision-focused learning, differentiable/submodular
    selection, budgeted adaptive inference, and edge-cloud offloading.
  - Verify: every factual related-work statement links to a primary paper.

## Checkpoint A

- [x] Continue only if the oracle/decomposition shows recoverable Fixed-byte
  regret concentrated near action/capacity decisions.

## Phase 2: Minimal frontier-aligned learner

- [x] Write failing tests for frontier targets and deployment contracts.
- [x] Implement the smallest shadow-price-aligned objective on the existing
  mean/max ActionValue encoder.
- [x] Run one seed x five OOF folds for OQ and optional OQR in parallel.
- [x] Compare against frozen CurrentActionValue under identical schedulers.

## Checkpoint B

- [x] Advance only if the frozen one-seed point gate in SPEC.md passes.
- [ ] If OQR fails but OQ passes, discard ROI rather than weaken the gate.

## Phase 3: One targeted refinement

- [ ] Diagnose the winning/failed screen by boundary stratum and action pair.
- [ ] Implement at most one evidence-backed refinement:
  `DeltaFrontier` for cross-action inconsistency or `SetValuedFrontier` for
  ensemble ambiguity, never both without an intermediate gate.
- [ ] Re-run the one-seed screen.

## Phase 4: Confirmation or stop

- [ ] Expand `OQR/BoundaryPathDistill` and its seed-matched OQ baseline to three
  seeds x five OOF folds.
- [ ] Evaluate frozen calibration once only after the OOF gate passes.
- [ ] Write Chinese results with facts, inferences, judgments, limitations,
  novelty evidence, and the next go/no-go decision.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Repeating decision-focused knapsack literature | Search before naming novelty; define only the task-specific difference supported by evidence. |
| Reward contexts leak across folds | Reuse only the completed fold-contained reward artifact and assert IDs. |
| ROI creates false gains | Compare OQ and OQR under the same gate; report ROI selection and harmful-action rate. |
| CPU oversubscription | One torch thread/job; bounded process pool; resumable checkpoints. |
| 90-minute window expires | Risk-first stages; stop weak branches early and preserve partial artifacts. |
