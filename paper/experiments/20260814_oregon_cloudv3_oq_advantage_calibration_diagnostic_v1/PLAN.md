# Implementation Plan

## Phase 1: Contracts

- Add the frozen protocol and failing pure-function tests.
- Verify tests fail because the calibration implementation is absent.

## Phase 2: Pure calibration

- Implement Q-anchor and midpoint O/Q advantage replacement.
- Implement the Q90-preferred diagnostic transformation.
- Verify the focused unit tests pass.

## Phase 3: Read-only diagnostic runner

- Load source OOF scores for all five folds and three seeds.
- Fit per-action and direct-advantage isotonic models from fit OOF only.
- Apply fixed methods to calibration and final full-fit scores.
- Freeze one global O/Q upgrade margin from fit-OOF candidates using calibration Official AP only.
- Reuse the exact source scheduler, MAUR predictions, and evaluator.

## Phase 4: Execution

- Sync only the isolated experiment directory to the server.
- Run at low CPU priority without changing the active final-completion job.
- Persist results, selections, calibrators, source hashes, and a Chinese report.

## Verification

- Unit tests pass locally and on the server.
- Source completion and Cloud-V3 hashes match the frozen protocol.
- Exactly 36 point rows are produced: 2 roles x 6 methods x 3 budgets.
- Every selection respects capacity and every calibrator reports fit-OOF-only provenance.
