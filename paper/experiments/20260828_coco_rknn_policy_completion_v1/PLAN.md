# Execution plan

1. Freeze the specification and source hashes.
2. Add failing contract tests for a true three-action CAVS-OQ export and a
   three-seed external ensemble export.
3. Implement and test the isolated CAVS-OQ training/evaluation/export path.
4. Recompute RKNN-domain ORIC rewards and features from fit only.
5. Train and export three-seed ORIC/MORIC with PyTorch/NumPy parity tests.
6. Generate point results without using final data for selection.
7. Copy only verified exports into the final-v2 bundle, recalibrate controls on
   the 571-image calibration role, rebuild the bundle manifest, and run smoke
   tests before formal hardware execution.

