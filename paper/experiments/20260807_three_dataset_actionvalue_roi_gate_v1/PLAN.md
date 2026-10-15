# Implementation plan

## Architecture decisions

- Create one isolated experiment; all completed experiments remain read-only.
- Reuse frozen OQ/OQR scores where they already exist.
- Train only the missing ENA24/Oregon four-action OQR learner.
- Select all AdvantageGate thresholds from fit OOF and evaluate calibration once.
- Run one resumable CPU process per dataset so datasets can execute concurrently.

## Tasks

### Phase 1 - contracts

- [ ] Add protocol and result-matrix validation.
  - Acceptance: dataset roles, canonical run names, actions, methods, modes, and
    budgets cannot drift.
  - Verify: `python -m unittest discover -s tests -v`.

- [ ] Add pure ROI advantage masking and OOF threshold selection.
  - Acceptance: OQ scores are unchanged; ROI can be disabled; threshold choice
    obeys the harmful-rate constraint and deterministic tie breaks.
  - Verify: focused synthetic tests fail before implementation and pass after.

### Phase 2 - data and training

- [ ] Add source preflight plus four-action stats/contextual rewards.
  - Acceptance: only fit/calibration sources are read; ROI reward is present
    whenever ROI is available.
  - Verify: remote preflight and 32-image smoke run.

- [ ] Add ENA24/Oregon OQR OOF/full training and scoring.
  - Acceptance: 5 folds x 3 seeds complete on CPU and produce finite aligned
    four-action scores; old OQ scores remain frozen inputs.
  - Verify: smoke model fits and predicts all four actions.

- [ ] Add COCO frozen-score reuse.
  - Acceptance: source OQ/OQR selection reproduces the completed diagnostic
    before applying the new learned gate.
  - Verify: source hash and selection reproduction checks.

### Phase 3 - evaluation and launch

- [ ] Add OOF threshold fitting, frozen calibration evaluation, and reporting.
  - Acceptance: 36 rows per dataset, capacity safety, ROI risk metrics, and OQ
    deltas are present.
  - Verify: result-matrix validator.

- [ ] Sync only the new experiment and launch three dataset jobs concurrently.
  - Acceptance: tests/preflights/smoke pass, PIDs and logs are saved, no GPU is
    used, and each process is resumable.
  - Verify: one immediate process/log health check, then stop monitoring.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Repeating the completed COCO OQ/OQR run | Reuse frozen scores; run only the new learned gate evaluation |
| OOF/calibration leakage | Fit thresholds on OOF only and store provenance with every calibration row |
| Oregon Cloud-V2 contamination | Whitelist `canonical_oregon_event_v1` and reject Cloud-V2 paths |
| ROI unavailable for some samples | Preserve the canonical action mask; never synthesize availability |
| CPU oversubscription | Separate dataset processes with bounded cache/training worker pools |
| Partial long run | Atomic checkpoints for stats, rewards, folds, models, scores, and evaluation |

## Checkpoints

- [ ] Contract tests green.
- [ ] All three preflights and smoke tests green.
- [ ] Three background tasks launched with distinct output roots.
- [ ] Final matrix and interpretation generated after completion.
