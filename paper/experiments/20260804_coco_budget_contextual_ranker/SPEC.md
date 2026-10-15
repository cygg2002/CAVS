# Spec: COCO Budget-Conditioned Contextual Ranker Diagnostic

## Objective

Determine whether a fit-only, budget-conditioned listwise ranker can outperform
the paper-faithful ORIC/MORIC reimplementations on the frozen COCO protocol.
The first phase reuses the exact ORIC AP50 contextual rewards and top-25 edge
features so that only the learning objective changes. AP50-95 rewards are
generated only if the cheaper objective diagnostic shows a positive signal.

## Tech Stack

- Python 3, NumPy, PyTorch, standard-library `unittest`.
- Frozen inputs and evaluator from
  `20260720_frozen_detector_multiranker_contextm2e`.
- Frozen ORIC/MORIC fit artifacts and checkpoints from
  `20260728_three_dataset_dcsb_oric_moric_external_baselines`.

## Commands

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
python3 scripts/run_diagnostic.py --stage calibration
python3 scripts/run_diagnostic.py --stage final
```

`final` is permitted only when the calibration gate written by the first
command passes.

## Project Structure

```text
configs/protocol.json      Frozen budgets, seeds, model and gate thresholds
scripts/ranker_core.py     Pure ranking, training and score utilities
scripts/run_diagnostic.py  Read-only source loading and isolated outputs
tests/                     Unit and protocol-contract tests
runs/                      New artifacts only
notes/                     Chinese result summary
```

## Code Style

Use typed, small functions and deterministic stable ordering:

```python
order = np.argsort(-np.asarray(scores), kind="mergesort")
```

No new framework or dependency is added.

## Testing Strategy

- Unit-test exact budget counts, deterministic pairs, budget-dependent scores,
  ensemble averaging, and calibration/final gate isolation.
- Smoke-test training on synthetic separable data.
- Run the completed external-baseline test suite as a regression check.
- Verify that source hashes are unchanged before and after every stage.

## Boundaries

- Always: train only on `fit`; use calibration only for the frozen go/no-go;
  compare Original payload at fixed-call 10%, 25%, and 40%.
- Ask first: changing detector predictions, dataset roles, payload definitions,
  or the final success thresholds.
- Never: train on calibration/final labels; inspect final before the gate;
  overwrite any completed experiment; tune after final is evaluated.

## Success Criteria

### Calibration gate

- Candidate mean AP gain over the better of ORIC/MORIC is at least 0.002.
- No fixed-call budget loses more than 0.001 AP.
- All configured seeds train and produce finite scores.

### Final target

With Original payload and no greater request KB/image, exceed the frozen
external results at all three points:

- 10%: 0.4493
- 25%: 0.4893
- 40%: 0.5191

Final is evaluated once, regardless of outcome.

## Open Questions

None for Phase A. A set encoder and AP50-95 reward are explicitly out of scope
until the objective-only diagnostic is known.
