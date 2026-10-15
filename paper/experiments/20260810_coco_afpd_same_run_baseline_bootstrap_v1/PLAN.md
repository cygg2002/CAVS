# Plan: AFPD Same-Run Baseline Bootstrap v1

## Architecture decisions

- Isolate the work in a new experiment directory; do not patch the running
  parent.
- Reuse parent final selections and canonical mixed-prediction evaluator.
- Gate execution on the parent's `runs/bootstrap/completion.json` marker.
- Limit scope to six Fixed-byte comparisons.

## Tasks

- [ ] Freeze protocol and pure contracts.
  - Acceptance: exact methods, budgets, replicate count, and post-hoc status.
  - Verify: unit tests reject any altered contract.
- [ ] Implement read-only preflight and bootstrap runner.
  - Acceptance: six resumable comparison artifacts plus one aggregate report.
  - Verify: local tests and remote preflight pass.
- [ ] Queue remote execution after the parent completes.
  - Acceptance: waiting process consumes negligible CPU and starts only after
    the parent completion marker appears.
  - Verify: launcher log records waiting/running state and parent code hashes.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| CPU contention with current bootstrap | Delays both experiments | Hard parent-completion gate before worker pool creation |
| Mixing old and retrained ActionValue | Invalid attribution | Only same-run `Current-OQ/OQR` keys are accepted |
| val2017 reuse overstated as confirmation | Misleading claim | Hard-code post-hoc/non-confirmatory flags |
| Interrupted bootstrap | Lost time | Save each comparison atomically and resume by file |
