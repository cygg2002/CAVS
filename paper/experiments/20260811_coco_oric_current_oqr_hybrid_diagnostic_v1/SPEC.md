# Spec: COCO ORIC–Current OQR Hybrid Diagnostic v1

## Objective

Determine whether a frozen ORIC image-value anchor combined with the frozen
Current action-advantage head can outperform the complete Current joint
scheduler on canonical COCO val2017. This is a cache-only, post-hoc mechanism
diagnostic; it is not a new hidden-final claim.

The hybrid score is frozen before evaluation:

```text
H(i, original) = ORIC(i)
H(i, a) = ORIC(i) + Current(i, a) - Current(i, original)
```

Thus ORIC supplies the value of calling the image, while Current supplies only
the within-image Q90/ROI advantage. No scale, coefficient, threshold, or
comparator is fitted on val2017.

## Methods

- `ORIC-Current-OQ`: ORIC anchor plus frozen Current-OQ advantages.
- `ORIC-Current-OQR`: ORIC anchor plus frozen Current-OQR advantages.
- `Current-OQ`: frozen three-seed Current-OQ ensemble.
- `Current-OQR`: frozen three-seed Current-OQR ensemble.

## Protocol

- Dataset: canonical COCO train2017 fit=9,678, calibration=571,
  val2017 final=5,000.
- Primary mode: exact Fixed-byte at 10%, 25%, and 40% of all-Original bytes.
- Auxiliary mode: Fixed-call at 10%, 25%, and 40%.
- Metric: Official AP50-95, AP50, and AP75 at confidence 0.001.
- Bootstrap: 1,000 paired image resamples for four frozen comparisons per
  budget.
- Detector inference: none.
- Selector retraining or tuning: none.

## Commands

```bash
python3 -m unittest discover -s \
  paper/experiments/20260811_coco_oric_current_oqr_hybrid_diagnostic_v1/tests -v

python3 \
  paper/experiments/20260811_coco_oric_current_oqr_hybrid_diagnostic_v1/scripts/run_experiment.py \
  --stage all
```

## Project Structure

```text
configs/protocol.json   Frozen protocol and source paths
scripts/core.py         Pure score-composition and protocol contracts
scripts/run_experiment.py  Source-bound evaluation/bootstrap/report runner
tests/test_core.py      Unit contracts
runs/                   Generated artifacts only
notes/RESULTS_ZH.md     Generated Chinese report
```

## Code Style

Use small typed functions, explicit image-ID alignment, deterministic sorting,
and atomic result writes. Reuse frozen source evaluators and schedulers instead
of copying them.

## Testing Strategy

- Unit-test that Original equals the ORIC anchor exactly.
- Unit-test that Q90/ROI differences equal Current action advantages exactly.
- Reject non-finite, misaligned, or missing-Original scores; preserve a
  legitimately unavailable Q90/ROI action as absent rather than fabricating it.
- Assert all Fixed-byte selections remain within capacity.
- Assert 24 aggregate rows and 12 bootstrap comparisons exactly.
- Reproduce frozen Current-OQ/OQR Fixed-byte results within 1e-9 AP.
- Reproduce frozen ORIC Original-only Fixed-call control within 1e-9 AP.

## Boundaries

### Always

- Reindex every score by canonical final image order.
- Pin and hash every frozen source artifact.
- Report Fixed-byte and Fixed-call separately.
- Label all outputs `posthoc_final_reuse_hybrid_diagnostic`.

### Ask First

- Retraining ORIC or Current.
- Adding a learned calibration or mixture coefficient.
- Changing the action space or detector cache.

### Never

- Tune on val2017.
- Modify a completed source experiment.
- Re-run detector inference.
- Call this a new registered or hidden final.

## Success Criteria

1. Four methods × two modes × three budgets produce exactly 24 unique rows.
2. All source-reproduction and capacity checks pass.
3. Twelve paired-bootstrap comparisons complete with 1,000 replicates each.
4. The report answers whether ORIC membership, ROI, or their combination changes
   Current performance without conflating budget modes.

## Open Questions

None. The hybrid equation and evidence boundary are frozen above.
