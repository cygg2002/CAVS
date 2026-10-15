# COCO Current-OQR Strict Attribution and CPU Profile v1

## Objective

Produce one isolated, post-hoc COCO experiment that answers five questions:

1. How strong is the Current learner as a pure image ranker when every selected
   image is forced to use Original under Fixed-call?
2. How does Current-OQ compare with frozen ORIC/MORIC and Confidence systems
   under the common Fixed-byte capacity?
3. What is the complete Current-OQR Fixed-byte result under the same retraining
   run?
4. What AP change is attributable to adding ROI within the seed-matched Current
   learner?
5. What are the parameter count, model-only CPU latency, scheduler latency, and
   analytic linear-layer GFLOPs of one Current-OQR model and its frozen
   three-seed ensemble?

This experiment does not create a new hidden final. COCO val2017 has already
been consumed and every result must be labelled `posthoc_final_reuse`.

## Frozen protocol

- Dataset: canonical COCO train2017 fit=9,678, calibration=571, val2017=5,000.
- Detector and action caches: canonical Cloud-V1 sources only.
- Metric: Official AP50-95, with AP50 and AP75 reported.
- Budgets: 10%, 25%, and 40%.
- Primary mode: Fixed-byte relative to all-Original payload capacity.
- Ranking control: Fixed-call Original-only.
- Current seeds: 20260809, 20260810, and 20260811.
- Current-OQ actions: Edge, Original, Q90.
- Current-OQR actions: Edge, Original, Q90, ROI.
- External and Confidence rows are frozen imports from the completed full
  scheduler benchmark. They are not selected or repaired using this result.

## Runtime profile contract

- Device: CPU.
- PyTorch intra-op threads: 1.
- Model is in `eval()` mode and inference uses `torch.inference_mode()`.
- Report batch=1 latency as median and p95 after warm-up.
- Report representative batched throughput separately.
- Report the capacity scheduler separately from model inference.
- Analytic GFLOPs count Linear multiply and add as two FLOPs. Embedding lookup,
  LayerNorm, SiLU, masking, Python feature construction, JPEG encoding, Edge
  detector, cloud detector, and network transfer are excluded and must be
  disclosed.
- Ensemble execution is serial on one CPU thread, matching the conservative
  deployment interpretation; no parallel speedup is assumed.

## Project structure

```text
configs/protocol.json    Frozen counts, budgets, sources, and profile settings
scripts/core.py          Pure validation, selection, and FLOP helpers
scripts/run_experiment.py Source-bound evaluation and CPU profiler
tests/test_core.py       Contract tests
runs/                    Generated results only
```

## Commands

```bash
python3 -m unittest discover -s \
  paper/experiments/20260810_coco_current_oqr_strict_attribution_profile_v1/tests -v

python3 \
  paper/experiments/20260810_coco_current_oqr_strict_attribution_profile_v1/scripts/run_experiment.py \
  --stage all
```

## Boundaries

### Always

- Reindex every score matrix against frozen image IDs.
- Verify source checkpoint and result hashes before evaluation.
- Keep single-seed rows separate from the three-seed ensemble.
- Use the same payload capacity for OQ and OQR.
- Distinguish source-faithful Original-only external rows from controlled
  SharedAction-NoROI rows.

### Never

- Retrain a detector or selector.
- Tune a threshold, choose a comparator, or repair a method on val2017.
- Modify any completed source experiment.
- Call the result a new registered or hidden final.
- Attribute the full Current-OQR versus external delta to ROI or pure ranking.

## Success criteria

- Exactly 24 newly evaluated Current rows: two Current methods x four score
  variants (three seeds plus ensemble) x three budgets for Fixed-byte.
- Exactly 24 Original-only Fixed-call Current ranking rows with the same matrix.
- All Fixed-byte selections remain within capacity.
- The ensemble Fixed-byte Current-OQ/OQR rows reproduce the frozen source result
  within 1e-9 AP.
- Frozen ORIC/MORIC and Confidence comparisons cover every requested work point.
- Parameter counts match checkpoint tensors and all three model architectures.
- CPU timing includes warm-up, repeated samples, median, p95, and environment.
- A Chinese report separates verified facts, attribution, and deployment caveats.

