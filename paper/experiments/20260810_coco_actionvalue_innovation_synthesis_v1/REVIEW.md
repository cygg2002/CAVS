# Review: COCO AFPD v1-v6 exploration

## Context

The changes are isolated experimental artifacts. They do not modify completed
canonical experiments, detector code, source caches, or the running full
COCO/Oregon benchmark.

## Correctness

- All six protocols keep calibration, audit, val2017, and registered final
  sealed for this exploration.
- Action columns are resolved per image from `action_id`; missing Q90/ROI
  actions are supported.
- Fixed-byte rows assert exact capacity safety.
- v5's go/no-go thresholds were written before the diagnostic result.
- v6's majority rule and screen gate were written before evaluation.
- One review finding was fixed: mandatory Original fallback retention was
  initially labeled as frontier consensus. The final report now separates
  `voted_frontier_consensus` from `retained_by_policy`; the prior report is
  archived under `runs/evaluation_pre_audit_semantics_fix/`.

## Readability and architecture

- Each mechanism has its own SPEC, PLAN, protocol, tests, and run artifacts.
- Existing AFPD loaders, evaluator, and scheduler are reused rather than
  copied or modified.
- No new dependency was added.
- v6 is intentionally a single experiment runner; further abstraction is not
  warranted because the branch failed its gate.

## Security

- No credentials or network input enter experiment logic.
- Source artifacts are treated as frozen local inputs.
- No user-provided strings are executed or interpolated into subprocesses.

## Performance

- Evaluation uses 12 forked CPU workers and completed in about 67 seconds once
  caches were loaded.
- The server had sufficient RAM; no detector inference or GPU was used.
- The independent long-running benchmark was not stopped or modified.

## Verification

- Local unit tests: 40/40 passed across v1-v6.
- v5/v6 Python compilation passed.
- v5 and final v6 artifacts were synchronized from the server.
- Local/server hashes were matched for v5 before synthesis; final v6 source
  and result hashes are recorded below.

Final v6 hashes:

```text
protocol  e68b0a6b5997806284ea4d51b3ac7c083ed55c37bdc71fced2ca56b5a8fb2146
core      5fea7c6eee08f1db85b40a05cf36049d78b183c9eebedaf1852df6a692b38f9c
runner    573af0a627129edf10194b51b2e75e94291035b5b949f856a542474b603a73d6
report    ac2345f4eaef37551518ec8f723461a1409ebcf8ef369cabdcb2789910b79234
results   42170525df37292431285d4295b3b020b24ab23820b24f066a59f7e1925378cc
```

## Limitations

- This is fit-only OOF evidence, not an independent paper result.
- AFPD has no paired bootstrap comparison against official ORIC/MORIC yet.
- Generic decision-focused learning, ensembles, and uncertainty sets are prior
  art; exact-method novelty remains an evidence-bounded literature claim.
- `ruff` was unavailable locally; unit tests and Python compilation were used.

## Verdict

**Approve as a reproducible exploratory result.** Freeze AFPD v1 for the next
independent evaluation. Do not promote v2-v6 or make paper-level superiority
claims from this development evidence.

