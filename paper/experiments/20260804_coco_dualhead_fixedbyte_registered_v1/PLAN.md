# Implementation Plan: COCO DualHead Fixed-Byte Registered Evaluation v1

## Architecture Decisions

- Reuse the frozen development data and model definitions read-only.
- Allocate only new evaluation roles; do not rebuild contextual rewards.
- Keep final inference itself behind the audit gate.
- Gate only DualHead against external ORIC/MORIC; ablations are descriptive.

## Tasks

### Phase 1: Contracts

- [ ] Write failing tests for exclusions, exact roles, fixed-byte comparator/gate,
  capacity safety, and one-time final access.
- [ ] Implement the minimal pure core and frozen protocol.
- Verify: isolated unit suite passes.

### Phase 2: Training and evaluation

- [ ] Load/hash the frozen development source and train seven methods.
- [ ] Implement common fixed-call/fixed-byte evaluation and audit freeze.
- Verify: synthetic contracts and complete result-matrix checks.

### Phase 3: Inference and final

- [ ] Add resumable audit inference and audit-gated final inference.
- [ ] Add paired bootstrap and final gate.
- Verify: local/remote tests and two-image detector smoke.

### Phase 4: Automation

- [ ] Prepare and audit the new roles.
- [ ] Launch a locked background pipeline with resumable stages.
- Verify: PID, log, first progress marker, and no final access before audit.

## Risks

| Risk | Mitigation |
|---|---|
| GPU contention | batch=1 cloud, resumable chunks, memory-aware retry |
| Accidental final access | explicit open-final marker after passed audit |
| Post-hoc comparator selection | freeze strongest external method on audit |
| Development OOF gain does not transfer | registered audit gate stops final |
