# Implementation Plan: COCO BudgetSet Registered Paper Evaluation v1

## Architecture Decisions

- Build a new additive experiment; do not modify completed results.
- Generate Original Edge/Cloud predictions once and reuse immutable chunks.
- Reuse the audited ORIC/MORIC, BudgetSet, COCO evaluator, and CostEligible
  primitives through explicit read-only imports.
- Keep audit and final as separate commands with completion gates.

## Task List

### Phase 1: Protocol contracts

- [ ] Write failing tests for deterministic disjoint roles, forbidden reward
  contexts, exact budgets, comparator freezing, and one-time final access.
  - Acceptance: tests fail because `paper_core` does not exist.
  - Verify: focused `unittest` command from repository root.
  - Files: `tests/test_paper_core.py`, `tests/test_stage_contracts.py`.

### Phase 2: Pure core

- [ ] Implement split, record, dual-metric reward, selection, gate, and bootstrap
  contracts.
  - Acceptance: all synthetic tests pass without GPU or COCO files.
  - Verify: new unit suite.
  - Files: `scripts/paper_core.py`.

### Checkpoint: contracts

- [ ] New tests pass and no completed experiment was edited.

### Phase 3: Resumable detector cache

- [ ] Implement prepare and chunked Original inference.
  - Acceptance: a two-image remote smoke passes for both frozen detectors and a
    repeated command skips verified chunks.
  - Verify: smoke completion hashes and prediction audit.
  - Files: `scripts/infer_original.py`, `scripts/run_paper.py`.

### Phase 4: Development and audit

- [ ] Build development-only contextual rewards and train all registered methods.
  - Acceptance: no audit/final ID occurs in reward rows or contexts.
  - Verify: reward isolation audit and checkpoint hashes.
- [ ] Evaluate audit and freeze per-work-point external comparators.
  - Acceptance: complete 5 x 2 x 3 result matrix and deterministic go/no-go.
  - Verify: audit completion and report hashes.

### Checkpoint: audit gate

- [ ] Open registered final only if the frozen audit gate passes.

### Phase 5: Registered final

- [ ] Evaluate final exactly once and run 1,000 paired-bootstrap replicates.
  - Acceptance: complete point estimates, intervals, capacity checks, and a
    registered success/failure result.
  - Verify: final completion hashes and rerun refusal.

### Phase 6: Review and report

- [ ] Perform five-axis code/protocol review and write Chinese results.
  - Acceptance: no unresolved high-severity issue; claims match registered gate.
  - Verify: new and regression tests plus artifact cross-check.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| GPU contention/OOM | inference delay | resumable chunks, small configurable batch |
| Reward generation is CPU-heavy | long runtime | compute AP50/AP50-95 in the same evaluations and checkpoint rows |
| New train2017 subset differs from val2017 | limited external validity | label exact population; do not mix with old val2017 numbers |
| Audit gate fails | final remains sealed | report a valid negative audit; do not tune and reopen |
| Final AP gain is not significant | paper claim fails | report point estimates and bootstrap gate honestly |
| Baseline adaptation criticism | unclear fairness | include paper AP50 and metric-matched AP50-95 ORIC/MORIC controls |
