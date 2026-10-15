# Implementation plan

- [ ] Validate the three RKNN-FP16 candidate roles and their frozen hashes.
- [ ] Refit DCSB and Adaptive Feeding on fit only and export pickle-free NumPy
      policy parameters.
- [ ] Prove host scikit-learn and board NumPy scores agree on held-out records.
- [ ] Freeze calibration-only empirical score references and 10/25/40 byte
      controls; freeze 100% zero-price controls for the two native rows.
- [ ] Extend an isolated copy of the final-v2 board runtime with the four new
      method identifiers and regenerate its hash manifest.
- [ ] Run all host and board unit tests, then two 50-frame online smoke tests.
- [ ] Execute the eight formal runs serially with new run IDs and preserve the
      completed 46-run archive unchanged.

