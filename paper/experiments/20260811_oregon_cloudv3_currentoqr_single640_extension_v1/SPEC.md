# Spec: Oregon Cloud-V3 CurrentActionValue Single-ROI Completion

## Objective

Complete the missing Oregon action-space comparison under the already frozen
Cloud-V3 protocol:

- `CurrentActionValue-OQ`: Edge / Original / Q90 (frozen source result);
- `CurrentActionValue-OQR-Current640`: Edge / Original / Q90 / `current@640`
  (new, retrained from scratch);
- `CurrentActionValue-OQR12`: Edge / Original / Q90 / twelve Profile x Scale
  ROI candidates (frozen source result).

The new single-ROI method is deliberately identical to the COCO ROI definition
so that the cross-dataset comparison is interpretable.

## Commands

```bash
python -m unittest discover -s tests -v
python scripts/run_experiment.py --stage preflight
python scripts/run_experiment.py --stage all
```

## Project Structure

- `configs/protocol.json`: frozen source, method, split, seed, and statistic contract.
- `scripts/core.py`: pure protocol/result validation.
- `scripts/run_experiment.py`: resumable train/evaluate/bootstrap pipeline.
- `tests/`: small contract tests.
- `runs/`: generated artifacts only.

## Code Style

Use deterministic, experiment-local Python. Import the completed Cloud-V3
experiment read-only; never patch or write its files.

## Testing Strategy

- Unit-test the exact four-action space and 18-job matrix.
- Assert every OQR selection belongs to the frozen four-action set.
- Assert Fixed-byte capacity is never exceeded.
- Assert final is prediction/evaluation only and never a training target.
- Run 1,000 paired event-primary and image-sensitivity bootstrap replicates for
  OQR versus OQ and OQR12 at all three budgets.

## Boundaries

- Always: retrain on Cloud-V3 fit only; retain the canonical event folds; save
  score/image-ID alignment and input hashes.
- Ask first: changing the single ROI away from `current@640`.
- Never: modify the source Cloud-V3 experiment, reuse an old learner checkpoint,
  tune on final, or label the reused final as hidden/confirmatory.

## Success Criteria

1. `3 seeds x (5 OOF folds + 1 full fit) = 18` new training jobs complete.
2. Calibration and final each contain six point rows (two modes x three budgets).
3. The final comparison table contains OQ, OQR-Current640, and OQR12.
4. Twelve bootstrap comparisons complete (2 comparators x 3 budgets x 2 units).
5. Source experiment hashes remain unchanged and no process writes below it.

## Open Questions

None. `current@640` is frozen as the strict COCO-matched single ROI.
