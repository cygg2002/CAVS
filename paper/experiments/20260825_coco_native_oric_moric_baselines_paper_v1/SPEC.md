# Spec: COCO Native ORIC/MORIC Baseline Completion

## Objective

Complete the COCO paper comparison with the scalar, Original-upload setting
used by ORIC/MORIC before any OQ/OQR action-space extension. Evaluate
`ORIC-Original`, `MORIC-Original`, `Confidence-Original`, and
`Random-Original` against the frozen `CAVS-OQR` system result.

This is a system-level comparison with different action spaces. The existing
OQR action-matched experiment remains the mechanism-control table.

## Protocol

- Domain: frozen PyTorch paper cache; RKNN/RK3588 artifacts are forbidden.
- COCO roles: fit=9,678, calibration=571, final=5,000.
- Native baseline actions: Edge or Original only.
- Budgets: Fixed-byte and Fixed-call at 10%/25%/40%; Fixed-byte is primary.
- ORIC/MORIC: reuse the three current-protocol scalar models trained from
  Original AP50-95 targets. Only the deployment action map changes, so
  retraining the same scalar model would be redundant.
- Confidence: frozen confidence score from the same Edge detections.
- Random: 50 deterministic seeds; report the 50-seed AP mean.
- Statistics: three-seed ORIC/MORIC mean/SD, ensemble points, and 1,000 paired
  image-bootstrap replicates. Random is nested across all 50 policies within
  every image bootstrap sample.

## Commands

- Test: `python -m unittest discover -s tests -v`
- Preflight: `python scripts/run_experiment.py --stage preflight`
- Full pipeline: `python scripts/run_pipeline.py --start-now`

## Project Structure

- `configs/protocol.json`: frozen source hashes and statistical contract.
- `scripts/core.py`: pure job/action-space validation.
- `scripts/run_experiment.py`: preflight, points, bootstrap, report.
- `scripts/run_pipeline.py`: resumable low-priority remote launcher.
- `tests/`: protocol and action-space tests.
- `runs/`: experiment-local outputs only.

## Boundaries

- Always: use the current Cloud hash, split, evaluator and capacity scheduler.
- Always: label native methods as Original-only and separate them from OQR
  action-matched controls.
- Never: modify source experiments, retrain detectors, tune on final, or read
  deployment-domain artifacts.
- Never: call the project reimplementation an official ORIC/MORIC result.

## Success Criteria

1. Every native selection contains only `original`; non-selected images remain
   Edge implicitly.
2. Fixed-byte utilization is at least 98% at all three budgets.
3. Final contains 3 budgets x 2 modes for every method; ORIC/MORIC also contain
   all three individual seeds.
4. All 12 registered Fixed-byte comparisons finish with 1,000 image-bootstrap
   replicates; Random uses all 50 policies per replicate.
5. CAVS-OQR points remain exactly 0.4802771321 / 0.5329001188 / 0.5601780940.

## Open Questions

None. The user explicitly requested the native non-OQR baselines.
