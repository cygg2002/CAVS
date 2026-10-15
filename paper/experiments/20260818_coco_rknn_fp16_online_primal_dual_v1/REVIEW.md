# Final Review

## Correctness

- The stream engine opens only the current image. A structural ordering test
  proves that the current cloud request and commit finish before the next image
  is opened.
- Every controller decision is checked against exact rational prefix capacity;
  all calibration and final replays report zero prefix-budget violation.
- `buffer_frames=0`, `zero_lookahead=true`, and `global_plan=false` are validated
  in the frozen control and bound into the immutable run contract.
- Interrupted runs rebuild the adaptive price by replaying committed current-frame
  observations; uninterrupted and resumed decisions are identical in tests.

## Experiment Integrity

- Step size is selected only from the preregistered grid on the 571-image
  calibration split. The 5,000-image final split is not used for tuning.
- Policy, calibration records, base control, sequence, RKNN, and cloud model are
  hash-bound. The original 20260817 experiment was not edited.
- The local AP materialization is explicitly marked diagnostic and not presented
  as a new RK3588 end-to-end latency run.

## Security and Reliability

- The bearer token is read only from an environment variable and is not written
  into the run contract or bundle.
- Authenticated HTTP requests do not follow redirects. Cloud identity and model
  hash are checked before execution, and malformed responses fall back to Edge.
- A run directory is protected by a single-writer lock and committed JSONL is
  append-only and checked against sequence order on resume.

## Performance

- Adaptive control costs approximately 6.4--7.0 microseconds per frame in the
  Mac replay, far below one millisecond. RK3588 overhead still needs to be read
  from the formal board run.
- Formal budget runs must remain serial so NPU, network, and cloud contention do
  not corrupt latency measurements. This does not introduce decision buffering:
  every individual frame is still uploaded immediately when selected.

## Remaining Validation Boundary

The implementation and diagnostic AP are complete. Publication-grade hardware
numbers require a new 50-frame board smoke and three 5,000-frame RK3588--server
runs using the packaged controls. A failed or interrupted formal run must use a
new run ID rather than being mixed with the completed prefix.
