# Spec: COCO CurrentActionValue OQR12 Profile-Scale Extension

## Objective

Evaluate whether the Oregon four-Profile x three-scale ROI action library also
improves CurrentActionValue on canonical COCO. The comparison is:

- frozen historical `Current-OQ`;
- frozen historical `Current-OQR` (MAUR; reference only);
- new, fully retrained single-ROI
  `CurrentActionValue-OQR-Current640-Reversible`;
- new, fully retrained `CurrentActionValue-OQR12`.

The retrained single-ROI row is required because the historical COCO OQR row
used MAUR while OQR12 uses Reversible Fusion. It isolates action-library breadth
from the fusion change without requiring another detector cache.

## Commands

```bash
python -m unittest discover -s tests -v
python scripts/run_pipeline.py --stage preflight
python scripts/run_pipeline.py --stage all
```

## Project Structure

- `configs/protocol.json`: frozen canonical/model/action/resource contract.
- `scripts/core.py`: pure action and resource contracts.
- `scripts/run_pipeline.py`: resource-gated cache, reward, training, evaluation,
  and bootstrap orchestration.
- `tests/`: contract tests.
- `runs/`: generated artifacts only.

## Code Style

Use deterministic experiment-local Python and read completed experiments only.
Every expensive stage is resumable by content signature.

## Testing Strategy

- Validate exactly four Profiles x three image sizes.
- Validate fit/calibration/final counts and disjoint IDs.
- Validate every cache against the frozen Cloud SHA and action signature.
- Validate 5-fold OOF plus three full-fit seeds.
- Validate exact byte capacity, action counts, and paired bootstrap inputs.

## Boundaries

- Always: use canonical COCO fit=9,678, calibration=571, val2017=5,000; use
  the same Edge/Cloud/Q90/Original definitions as the completed COCO benchmark;
  use Reversible Fusion for every ROI candidate.
- Ask first: changing detector weights, split, reward, or fusion.
- Never: modify a completed experiment, tune on val2017, claim a hidden final,
  terminate another user's GPU process, or start inference below the frozen
  11 GB free-memory guard.

## Success Criteria

1. Twelve ROI actions exist for all three roles with the frozen Cloud hash.
2. Reversible single-ROI and OQR12 each complete 5-fold OOF and three full-fit
   seeds from scratch.
3. Fixed-byte and Fixed-call 10/25/40 report historical OQ/OQR and the two new
   Reversible methods under one evaluator.
4. 1,000-replicate paired bootstrap compares OQR12 with OQ and OQR.
5. The pipeline waits for three consecutive configured GPU resource checks;
   the 11 GB free-memory guard remains mandatory.

## Open Questions

None. This is a post-hoc extension because COCO val2017 has already been used.
