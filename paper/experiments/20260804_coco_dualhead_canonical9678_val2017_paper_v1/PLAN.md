# Implementation plan

1. Lock and validate the canonical source manifests and exact role counts.
2. Load Edge/Original predictions for fit, calibration and final from the same
   canonical cache used by the prior paper experiments.
3. Generate resumable AP50 and AP50-95 contextual rewards on fit=9,678 only.
4. Train all seven methods with the unchanged frozen model hyperparameters.
5. Freeze external fixed-byte comparators on calibration=571.
6. Evaluate the 42-point matrix on full val2017=5,000 and run paired bootstrap.
7. Audit hashes, role isolation and report provenance before marking complete.

