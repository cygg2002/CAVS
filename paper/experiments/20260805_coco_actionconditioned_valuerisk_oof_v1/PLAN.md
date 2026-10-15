# Implementation plan

- [x] Freeze source paths, hashes, fit/calibration populations and fold manifest.
  - Verify: fit=9,678; calibration=571; overlap=0; forbidden roles absent.
- [x] Add pure action-record, scheduler, quantile-loss and gate contracts.
  - Verify: failing tests first; unavailable actions excluded; one action/image;
    strict call counts and byte capacities hold.
- [x] Add CurrentDualHead, ActionValue and ValueRisk OOF training.
  - Verify: each target receives exactly one OOF prediction; fold checkpoints
    are source-fingerprint bound.
- [x] Evaluate OOF and frozen calibration with Official COCO metrics.
  - Verify: 3 methods x 2 modes x 3 budgets x 2 roles; source hashes unchanged.
- [x] Write Chinese report and apply the registered gate.
  - Verify: no audit/final access and explicit go/no-go decision.
