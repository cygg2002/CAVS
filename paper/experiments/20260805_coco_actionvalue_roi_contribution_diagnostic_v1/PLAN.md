# Implementation Plan

- [x] Add pure action filtering, Oracle-relative eligibility and gate contracts.
  - Verify: tests reject unavailable/suboptimal ROI and enforce the frozen gate.
- [x] Reuse frozen ActionValue score artifacts and canonical evaluators.
  - Verify: exact image ordering and source SHA-256 bindings.
- [x] Evaluate the three policies at fixed-byte 10%/25%/40%.
  - Verify: all capacities hold and action counts sum to selected images.
- [x] Check fixed-call selection invariance without rerunning duplicate AP jobs.
  - Verify: full, no-ROI and Oracle-gated selections are identical.
- [x] Write the Chinese decision report.
  - Verify: explicit retain/drop ROI decision and no forbidden role access.
