# Implementation Plan

1. Freeze Cloud-V3 best checkpoint provenance and the unchanged 30-epoch recipe.
   - Verify: protocol unit tests assert all critical fields.
2. Add the minimal validated Cloud-V4 launcher.
   - Verify: tests pass and `--preflight-only` validates server paths and hash.
3. Launch in a detached process without modifying Cloud-V3.
   - Verify: PID is alive, log contains the new run name, and GPU training starts.
4. After completion, compare the best of 30 epochs with Cloud-V3 epoch 28.
   - Verify: `results.csv` has 30 rows and both checkpoint files exist.

## Risks

- Current free GPU memory is close to Cloud-V3's observed 8.86 GB use. Preserve
  batch 8 for comparability; if allocation fails, wait for memory rather than
  silently changing the frozen recipe.
- A fresh stage resets optimizer and warmup. This is deliberate and matches the
  transition from Cloud-V2 best to Cloud-V3.
