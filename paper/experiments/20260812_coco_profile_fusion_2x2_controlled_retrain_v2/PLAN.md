# Implementation Plan

1. Freeze the corrected protocol and causal-control tests.
2. Reuse v1 action caches read-only and build one unified six-action reward map.
3. Retrain all four cells with the same folds, seeds, adapter and inner split.
4. Evaluate fit OOF, calibration and public val under Fixed-byte/Fixed-call.
5. Run the five factorial contrasts at three budgets with 1,000 paired bootstraps.
6. Write machine-readable artifacts and a Chinese report.

Verification checkpoints:

- Unit tests and protocol validation pass before preflight.
- Preflight hashes every source cache and rejects any reused learner artifact.
- A smoke preflight/reward manifest check passes before background launch.
- Background PID, log and launch metadata are saved under `runs/launcher/`.
