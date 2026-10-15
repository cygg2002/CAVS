# Spec: RKNN-FP16 CAVS Strict Online Primal-Dual Control

## Objective

Reduce the gap between globally scheduled CAVS-OQR and its causal online
deployment without introducing lookahead or buffering.  Each COCO image is
opened only after the previous image has completed Edge inference, decision,
any cloud request, MAUR fusion, and durable commit.

The new controller keeps the frozen three-seed RKNN-FP16 CAVS-OQR value model,
actions, codecs, cloud model, image order, and exact prefix-byte constraint.  It
only replaces the calibration-frozen constant byte price with a causal
projected primal-dual update.

## Assumptions

1. "Immediate upload" means the selected current-frame payload is sent before
   the next image is read; it does not mean asynchronous overlap or a frame
   window.
2. JPEG payload bytes remain the paper budget.  HTTP wire bytes are reported
   separately and do not control action admission.
3. The existing fixed-price controller is the registered baseline.  This is a
   new isolated experiment and does not modify its package or results.
4. Step size and any price bounds are selected using fit/calibration evidence
   only.  Final COCO ground truth is never used to tune the controller.

## Decision Rule

At frame `t`, after observing only the current image, Edge detections, action
values, and current payload byte sizes, admit only actions satisfying

`spent_before + bytes(t,a) <= floor(rho * original_prefix_through_t)`.

Among feasible actions choose

`argmax_a value(t,a) - lambda_t * bytes(t,a) / median_original_bytes`.

After committing that action, update

`lambda_{t+1} = clip(lambda_t + eta * (bytes(t,a) - rho * original_bytes_t)
                     / median_original_bytes, lambda_min, lambda_max)`.

`eta=0` must be exactly equivalent to the existing fixed-price controller.

## Commands

- Unit tests:
  `python -m unittest discover -s board_bundle/tests -p 'test_*.py'`
- Host tests:
  `python -m unittest discover -s host/tests -p 'test_*.py'`
- Causal replay:
  `python host/run_primal_dual_replay.py --help`
- Bundle verification:
  `python board_bundle/scripts/verify_bundle.py --manifest board_bundle/BUNDLE_MANIFEST.json --root board_bundle`
- Board execution:
  `bash board_bundle/run_online.sh --help`

## Project Structure

- `board_bundle/scripts/`: standalone strict sequential runtime and controller.
- `board_bundle/controls/`: calibration-frozen adaptive controls.
- `board_bundle/tests/`: pure controller and frame-order contract tests.
- `host/`: calibration/final replay, evaluation, packaging, and reports.
- `runs/`: generated replay and board artifacts.

## Code Style

Use deterministic plain Python with explicit immutable records and exact
`Fraction` budget arithmetic.  Keep the hot decision path allocation-light and
dependency-free beyond the existing NumPy runtime.

## Testing Strategy

- Unit: update direction, non-negative/bounded price, exact prefix feasibility,
  deterministic ties, `eta=0` parity, and resume equivalence.
- Integration: event trace proves `observe(t) -> decide(t) -> upload(t) ->
  commit(t) -> observe(t+1)` with no suffix access.
- Replay: compare fixed and adaptive controllers on the same frozen order,
  scores, payload costs, and action outputs.
- Hardware: fresh run IDs on RK3588; report decision overhead, AP, action counts,
  bytes, failures, latency percentiles, and throughput.

## Boundaries

- Always: preserve exact prefix-byte feasibility and bind all runtime artifacts
  by SHA-256.
- Ask first: changing model, action semantics, budget denominator, image order,
  or cloud protocol.
- Never: inspect a future image before committing the current frame, tune on
  final labels, reuse a previous run ID, or call replay results hardware
  deployment results.

## Success Criteria

1. `zero_lookahead=true`, `global_plan=false`, `buffer_frames=0` are recorded in
   every run contract.
2. Maximum prefix budget violation is exactly zero for 10%, 25%, and 40%.
3. `eta=0` reproduces every fixed-controller decision and serialized state.
4. A resumed adaptive run is identical to uninterrupted execution.
5. The selected payload is uploaded and resolved before the next image opens.
6. Adaptive decision overhead is measured and is below 1 ms/image on the host;
   board overhead is reported from the formal run.
7. Final AP is evaluated once only after the adaptive control is frozen.

## Open Questions

- Whether the calibration-selected adaptive controller improves final AP is an
  experimental result, not an assumption.  If it does not, the fixed controller
  remains the paper method.
