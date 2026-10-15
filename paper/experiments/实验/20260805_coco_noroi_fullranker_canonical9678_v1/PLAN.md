# Implementation Plan

1. Freeze the protocol, source experiments, method matrix, policies, budgets,
   worker counts, and post-hoc result status.
2. Add pure contract tests for the 192-row matrix, strict No-ROI policies,
   Fixed-byte capacity, score alignment, and final-isolation metadata.
3. Export canonical and historical frozen scores for calibration and val2017,
   proving image-ID alignment before evaluation.
4. Select and evaluate all Fixed-call and Fixed-byte points. Freeze lambdas and
   comparator identities using calibration only.
5. Run the paired bootstrap comparisons with multiprocessing and write a concise
   Chinese report.
6. Run preflight and unit tests locally, sync the independent experiment to the
   server, and launch the resumable pipeline with `nohup`.

