# Plan: COCO Relational Action-Value OOF Diagnostic v1

## Architecture decisions

- Keep the existing Original/Q90 joint scheduler and replace only the encoder
  and/or objective in a 2x2 factorial design.
- Build one fold-contained reward cache shared by every model and seed.
- Run risk-first: preflight and reward-isolation checks precede model training;
  OOF gate precedes the one-time calibration stage.
- Parallelize independent reward evaluations and model/seed/fold jobs on CPU;
  keep PyTorch threads per worker at one.

## Tasks

### Task 1: Freeze contracts and source bindings

- Acceptance: COCO fit/calibration only, Original/Q90 only, exact method matrix,
  folds, seeds, budgets, and gates are machine validated.
- Verify: protocol unit tests and remote source preflight.
- Files: `SPEC.md`, `PLAN.md`, `configs/protocol.json`, protocol tests.

### Task 2: Build fold-contained rewards

- Acceptance: every target has Original/Q90 rewards, target is absent from its
  context, every fit context stays in the target fold, and calibration contexts
  contain calibration IDs only.
- Verify: synthetic isolation tests plus generated reward audit.
- Files: `scripts/core.py`, `scripts/run_experiment.py`, reward tests.

### Checkpoint A

- All tests pass.
- Source hashes and role counts are recorded.
- No forbidden role has been opened.

### Task 3: Implement the 2x2 model family

- Acceptance: mean/max and relational encoders share the same action/value
  heads; global ranking toggles independently; relational output is invariant
  to valid-box permutation.
- Verify: RED/GREEN model, loss, permutation, finite-gradient smoke tests.
- Files: `scripts/core.py`, model tests.

### Task 4: Implement OOF training and evaluation

- Acceptance: 4 methods x 3 seeds x 5 folds complete exactly once; scores align
  with image IDs; six budget work points are evaluated per method/seed/ensemble.
- Verify: synthetic pipeline smoke and result-matrix assertions.
- Files: `scripts/run_experiment.py`, pipeline tests.

### Checkpoint B

- OOF scores contain no in-fold training predictions.
- Every saved selection is No-ROI and within capacity.
- The frozen OOF gate is recorded before calibration.

### Task 5: Conditional calibration and reporting

- Acceptance: calibration is skipped on OOF failure or evaluated once with
  full-fit frozen settings on OOF success; Chinese report explains which of
  representation/objective/full interaction passed.
- Verify: gate branch tests and completion metadata.
- Files: `scripts/run_experiment.py`, reporting tests.

### Task 6: Remote launch

- Acceptance: local and remote tests/preflight pass; background PID/log exist;
  CPU worker count is honored; source experiments remain unchanged.
- Verify: one process/status/log check after launch, then stop monitoring.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Context reward generation is long | Resumable JSONL checkpoints and multi-process workers |
| Attention model is CPU-heavy | Two small layers, mini-batches, one Torch thread per process |
| Cross-image loss creates quadratic memory | Compute within bounded mini-batches only |
| Calibration becomes an implicit tuning set | OOF-only gate and frozen one-time calibration branch |
| Results are confused with val2017 paper numbers | Explicit diagnostic role/result labels and forbidden-role flags |
