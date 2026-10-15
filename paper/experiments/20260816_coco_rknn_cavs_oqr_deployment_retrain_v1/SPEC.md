# Spec: RK3588 FP16 and INT8 CAVS-OQR deployment-aligned retraining

## Objective

Train two isolated CAVS-Current-MAUR policies from the exact one-image RKNN
candidate records produced on an Orange Pi 5 Ultra / RK3588: `fp16` and
`int8-heads`.  Each policy is evaluated only against candidates from the same
backend.  This measures deployment-aligned routing accuracy and cache-to-HTTP
parity; it does not replace the existing PyTorch paper-protocol result.

## Frozen Protocol

- COCO roles: `fit=9,678`, `calibration=571`, `final=5,000`.
- Actions: Edge, Original, Q90, and Current@640 with MAUR.
- Fixed-byte budgets: 10%, 25%, and 40% of native COCO JPEG bytes.
- Five frozen fit folds; seeds `20260809`, `20260810`, and `20260811`.
- Each backend has 18 learner jobs: five OOF folds plus one full-fit model for
  each seed.  No prior checkpoint is reused.
- The board generates candidates separately per model.  Host-side reward
  construction, learner training, and evaluation may run in parallel once the
  two validated candidate bundles are available.

## Commands

Board candidate collection is intentionally deferred until the existing board
bundle is mounted and its candidate JSON schema is inspected.  The generated
board command will be recorded in `PLAN.md` before it is run.

Host preflight and unit test:

```bash
cd /Users/cygg_macmini/work/Codex/ultralytics-main
python3 -m pytest -q \
  paper/experiments/20260816_coco_rknn_cavs_oqr_deployment_retrain_v1/tests
```

## Project Structure

- `configs/`: immutable protocol and candidate-manifest contracts.
- `scripts/`: candidate validation, import, training, evaluation, and export.
- `tests/`: compact regression tests for backend and role isolation.
- `runs/`: generated, resumable artifacts only.

## Code Style

```python
if candidate.backend != backend:
    raise RuntimeError("candidate backend does not match the training run")
```

Use explicit backend names, JSON/NPZ hash manifests, atomic writes, and
fail-closed validation.  Do not silently substitute FP16 records for INT8.

## Testing Strategy

- Unit tests validate allowed backend names, exact COCO role populations,
  complete job matrices, and backend/hash isolation.
- Candidate import tests will use small checked-in synthetic records before
  reading board artifacts.
- Preflight validates candidate count, unique image IDs, model hash, action
  schema, and source digest before any reward or learner job can start.
- Final validation reports cache AP and real HTTP AP separately for every
  backend and budget.

## Boundaries

- Always: preserve existing experiments; verify every candidate manifest;
  report FP16 and INT8 separately.
- Ask first: changing the cloud model, COCO split, action space, or budget
  denominator.
- Never: reuse a PyTorch candidate cache or learner checkpoint; combine FP16
  and INT8 candidates; use final images for training/model selection; place
  credentials or server URLs in tracked configuration.

## Success Criteria

1. Both board candidate bundles validate for all three COCO roles.
2. Each backend completes 18 new learner jobs and exports three new NumPy
   policies.
3. Each result reports AP50-95/AP50/AP75, exact byte usage, action counts,
   cache-versus-HTTP parity, and latency.
4. Neither model is called the primary deployment result unless its candidate
   cache and matching real HTTP run both complete without fallback errors.

