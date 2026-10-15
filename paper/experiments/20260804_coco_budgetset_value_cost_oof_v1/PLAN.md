# Implementation Plan: COCO BudgetSet Value/Cost OOF Diagnostic v1

## Architecture Decisions

- Add a standalone experiment and import completed primitives read-only.
- Keep the set representation fixed; isolate only target/cost coupling.
- Produce OOF predictions first, then evaluate every method with one common
  selection/evaluator path.

## Tasks

### 1. Pure contracts

- [ ] Write failing tests for folds, cost stripping, dual-head separation,
  diagnostics, and the frozen gate.
- [ ] Implement only the pure functions/model required by those tests.
- Verify: focused unit suite passes.

### 2. OOF training

- [ ] Load development records/rewards only and verify source hashes.
- [ ] Train Current-BudgetSet, SetValue, DualHead, and external controls for
  each fold without reading non-development artifacts.
- [ ] Save each completed fold atomically for resume.
- Verify: synthetic smoke and five fold files with disjoint target IDs.

### 3. Evaluation and report

- [ ] Assemble exactly one OOF prediction per image.
- [ ] Evaluate fixed-call/fixed-byte AP and ranking diagnostics.
- [ ] Apply the frozen go/no-go rule and generate the Chinese report.
- Verify: result matrix completeness, capacity safety, and source-boundary
  audit.

## Risks

| Risk | Mitigation |
|---|---|
| Repeated OOF neural training is slow | Resume by fold and use the available GPU; no detector inference |
| Contextual labels cross OOF folds | Disclose; use only as a development objective diagnostic |
| DualHead changes more than the objective | Keep the encoder dimensions and value loss identical to SetValue |
| Existing GPU workload causes OOM | The model/cache tensors are small; fall back to CPU automatically |
