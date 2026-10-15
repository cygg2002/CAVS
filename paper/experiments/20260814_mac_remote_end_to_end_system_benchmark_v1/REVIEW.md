# Review

## Verdict

Approved for an isolated exploratory system benchmark. It must not be presented
as the final RK3588 deployment result or as a new hidden COCO evaluation.

## Correctness

- The Edge and Cloud model hashes match the frozen deployment manifest.
- O/Q actions use Q95/Q90 JPEG and the existing exact-capacity scheduler.
- The formal run contains 100 unique images, 200 unique real requests, and zero
  request failures. All three plans stay within their raw-JPEG capacities.
- COCO category conversion and subset evaluation use the standard COCO metric.
- A failed cloud request explicitly preserves the Edge prediction.

## Readability and architecture

- The experiment is isolated under one dated directory and does not change the
  demo or paper experiment implementations.
- The benchmark reuses the deployed policy, protocol helpers, and scheduler.
- The instrumented service is intentionally separate from the live demo port.

## Security

- Both Cloud services bind to loopback and are reached through SSH tunnels.
- No credentials, passwords, or private keys are stored in artifacts.
- Request size and payload byte identity are validated before inference.
- The isolated service has no production authentication and must not be exposed
  on a public interface.

## Performance and interpretation

- Requests are sequential, so the benchmark does not create a GPU request burst.
- The remote GPU was shared with other jobs; measured Cloud/RTT values include
  contention and are not an idle-server lower bound.
- The diverse COCO image shapes trigger Mac MPS latency outliers. A fixed-shape
  30-run diagnostic after five warmups measured 15.34 ms median and 18.55 ms P95,
  while the diverse replay measured 16.77 ms median and 288.85 ms P95 for Edge
  inference. The latter is the observed replay cost, not a steady-camera claim.
- The reported blocking E2E is the sum of real component timings. It excludes
  global-window buffering delay and therefore remains comparable to a serial
  per-image latency model.

## Verification

- 24 focused and regression tests pass.
- Both benchmark scripts compile.
- 10-image smoke: 20/20 requests succeeded.
- 100-image formal run: 200/200 requests succeeded; artifact audit passed.
