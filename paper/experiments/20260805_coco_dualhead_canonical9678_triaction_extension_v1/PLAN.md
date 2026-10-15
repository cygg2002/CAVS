# Implementation plan

1. Validate the immutable source experiment, canonical role counts, action
   caches, action models, GBDT model, and frozen thresholds.
2. Export the seven calibration/final score matrices from the frozen canonical
   checkpoint and validate image ordering against the action cache.
3. Reproduce SharedAction, CostEligible-AllROI, and GBDT-Hierarchical action
   maps using the existing action engine.
4. Freeze per-policy external comparators and Fixed-byte lambda values on the
   571-image calibration role.
5. Evaluate the 126-point matrix on complete val2017 and save selections and
   mixed predictions needed by bootstrap.
6. Run 1,000 paired bootstraps for DualHead against each frozen external
   comparator at the nine primary policy/budget points.
7. Recheck all source hashes and generate a concise Chinese result report.

