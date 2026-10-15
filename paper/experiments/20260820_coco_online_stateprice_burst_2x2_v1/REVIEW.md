# Final Review

## Correctness

- 12/12 unit tests pass. They cover exact rational prefix capacity, monotone
  fixed-window pacing, exact final capacity, current-frame suffix invariance,
  reference-controller parity, invalid controls, and selection-cell binding.
- All three fixed+strict final action traces exactly match the frozen reference.
- Every final cell contains 5,000 decisions, zero envelope violations, and final
  bytes no greater than the registered absolute capacity.
- All controls are persisted before the final frame file or frozen-final
  reference selections are opened.

## Readability and Architecture

- The experiment is isolated from the 20260817/20260818 frozen packages.
- Controller logic accepts one current record rather than a sequence or window.
- The state price has one documented empirical inverse-demand rule; no learned
  policy framework or new dependency was introduced.

## Security and Data Integrity

- Every external calibration, policy, control, sequence, final log, and reference
  selection is SHA-256 checked before use.
- The local cloud bearer token was provided only through an environment variable
  and does not appear in source, configs, reports, or results.
- Cloud responses use the already reviewed authenticated client and exact model
  SHA-256 validation from the frozen 20260818 package.

## Performance

- Fixed-price decision overhead is approximately 4 microseconds/frame.
- The unoptimized 4,626-entry empirical state-price lookup costs 0.15--0.32
  milliseconds/frame on Mac. This is acceptable for diagnosis but should be
  compressed to a Pareto frontier with binary search before an RK3588 package.

## Statistical and Deployment Limits

- AP values are point estimates; no bootstrap has been run.
- 10% changed cloud actions were executed on Mac MPS while its source run used
  RTX. The model and inference protocol hash match, but it remains diagnostic.
- Materialization reuses unchanged real online outputs and executes only changed
  cloud actions. It does not create new end-to-end timing evidence.

## Verdict

Approved as a causal mechanism diagnostic. Not approved as a replacement
publication method or registered hardware result. The state-aware policy is not
robust across budgets; fixed-price paced/burst is simpler and directionally
positive but too small to claim without a fresh same-backend HIL run and paired
bootstrap.
