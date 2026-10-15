# Implementation Plan

1. Freeze Cloud-V4 source provenance and copy the Cloud-V4 training recipe.
   - Acceptance: exact source path/SHA and destination are in `protocol.json`.
   - Verify: protocol unit tests.
2. Add a minimal validated Cloud-V5 launcher.
   - Acceptance: import does not require Ultralytics; remote preflight verifies
     paths, O2M YAML, source SHA and frozen settings.
   - Verify: unit tests and `--preflight-only`.
3. Launch the run in a detached server process.
   - Acceptance: PID is alive, run directory is new, log shows Cloud-V5 and GPU
     training begins with batch 8.
   - Verify: process, log, run directory and GPU checks.
4. After the asynchronous run completes, compare all 30 metric rows with V4.
   - Acceptance: `best.pt`, `last.pt`, 30 result rows and best epoch are present.

## Risks

- A separate GPU job is using about 3.5 GB. Preflight available memory and
  launch only if the frozen batch 8 job fits; do not silently reduce batch.
- Fresh optimizer/warmup can initially reduce AP. This is deliberate and is the
  same stage semantics used for Cloud-V3 to Cloud-V4.
- Cloud-V4 and Cloud-V5 output names differ by source stage; assert isolation.
