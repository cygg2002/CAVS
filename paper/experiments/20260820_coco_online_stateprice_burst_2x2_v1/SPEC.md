# Spec: COCO Zero-Buffer State-Price × Budget-Envelope 2×2

## Objective

Measure how much of the CAVS-OQR online gap is reducible without buffering or
future-frame access. Compare two price policies and two byte-feasibility
envelopes on identical RKNN-FP16 current-frame candidates:

| Factor | Level 1 | Level 2 |
|---|---|---|
| Opportunity price | frozen fixed shadow price | calibration-fitted state-aware bid price |
| Byte envelope | exact observed-byte prefix | finite initial burst with the same final absolute capacity |

The experiment is diagnostic host replay first. It must not modify the frozen
20260817 or 20260818 experiments, and it must not be described as a new board
latency measurement.

## Assumptions

1. A formal deployment window has a known horizon and absolute byte capacity.
   For COCO these are 5,000 frames and the same 10%/25%/40% final capacities as
   the paper. A live service would use a fixed-duration byte window instead.
2. Reading the configured horizon and total capacity is allowed; reading future
   images, future scores, future action costs, or future labels is forbidden.
3. The current CAVS learner, action semantics, RKNN candidate domain, action
   values, and image order remain frozen.
4. Calibration predicted action value is the only tuning objective. COCO final
   annotations and final AP are evaluation-only.

## Controller Definitions

### Fixed price

Use the registered per-budget lambda and median byte normalization from the
frozen 20260817 control.

### State-aware price

Build an empirical price-demand curve solely from the 571 calibration records.
At frame `t`, compute remaining configured bytes divided by remaining frames,
then select the calibration price whose expected normalized demand best fits
that remaining rate without exceeding it. Select the current action by
`score - lambda_t * bytes / median_original_bytes`.

This policy uses only current candidates plus `(t, spent_bytes, total_capacity,
total_frames)` and a frozen calibration curve.

### Exact prefix envelope

At every frame enforce:

`spent_t <= floor(rho * cumulative_original_bytes_t)`.

### Finite-burst envelope

For final capacity `B`, horizon `T`, and initial burst `D`, enforce:

`spent_t <= floor(D + (B - D) * t / T)`.

This envelope is monotone, permits immediate early transmission, and equals
exactly `B` at `t=T`. The burst fraction is selected on calibration from:

`{0, 0.0025, 0.005, 0.01, 0.02, 0.05}` of final capacity.

## Commands

```bash
python -m unittest discover -s tests -v
python scripts/run_factorial.py --config configs/experiment.json
python scripts/summarize_factorial.py --runs runs --output RESULTS_ZH.md
```

AP materialization reuses the frozen evaluator and same-model cloud execution
from the 20260818 experiment only after final selections are frozen.

## Project Structure

- `scripts/`: isolated controller, calibration, replay, and reporting code.
- `tests/`: pure unit and causal-invariance tests.
- `configs/`: frozen source paths, hashes, grids, and selected controls.
- `runs/calibration/`: all calibration trials.
- `runs/final/`: one final replay per budget with four cells.
- `runs/materialized/`: derived AP evaluation views.

## Code Style

Python 3.12 standard library only for controller/replay logic. Use exact integer
or rational byte comparisons, deterministic action-order tie breaking, typed
dataclasses, atomic JSON output, and explicit input validation.

## Testing Strategy

- Unit tests: envelope boundaries, final-cap equality, inverse-demand price,
  tie breaking, and invalid configuration.
- Causal tests: identical observed prefixes produce identical decisions even
  when future suffixes differ.
- Regression test: fixed-price + exact-prefix reproduces frozen selections.
- Final audit: every cell has 5,000 unique decisions, zero envelope violation,
  and `final_used_for_tuning=false`.

## Boundaries

- Always: preserve `buffer_frames=0`, hash-bind sources, and report final byte
  capacity separately from prefix/burst semantics.
- Ask first: change the CAVS learner, action space, cloud model, image order, or
  final budget definition.
- Never: tune on final AP, read future-frame candidates during replay, reuse a
  mismatched backend without labeling it diagnostic, or call replay latency a
  real RK3588 system measurement.

## Success Criteria

1. Fixed+prefix reproduces the frozen 10%/25%/40% action traces exactly.
2. All four cells decide one current frame at a time with zero lookahead and no
   waiting; strict cells have zero exact-prefix violations and burst cells have
   zero pacing-envelope and final-cap violations.
3. Calibration freezes every parameter before final replay.
4. Changed-action AP is evaluated on all 5,000 images where backend-compatible
   outputs can be materialized.
5. The report separates predicted-value improvement, AP improvement, budget
   utilization, and protocol limitations.

## Open Questions

None for the diagnostic replay. A positive result will require a separate board
runtime package and fresh RK3588 HTTP runs.
