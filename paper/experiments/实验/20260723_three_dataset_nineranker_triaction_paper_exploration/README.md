# Three-Dataset Nine-Ranker TriAction Exploration

This is an isolated exploratory paper-level evaluation. It does not modify or
replace any canonical experiment.

Primary ninth Ranker: `Current-Controlled-32D`. Historical
`Legacy-Frozen-32D` is appendix-only and excluded from shared calibration.

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

Run one dataset:

```bash
python scripts/run_dataset.py --dataset DATASET --stage preflight
python scripts/run_pipeline.py --dataset DATASET
```

Each dataset writes only to `runs/{dataset}`. `CORE_COMPLETED` is written after
the 384-row main matrix, 120-row per-Ranker ablation, 1,200 Random seed rows,
canonical reproduction, and input-hash checks pass. `COMPLETED` is written only
after all 24 paired-bootstrap comparisons pass their integrity checks.

COCO and ENA24 remain diagnostic finals. Oregon is explicitly post-hoc reuse
of an already accessed registered final.
