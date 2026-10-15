# Spec: COCO AFPD + Occupancy Paper Protocol v1

## Objective

Resolve the attribution confound in the AFPD development result and evaluate
the frozen AFPD v1 and Occupancy Gate v2 under one canonical COCO protocol.
The experiment must distinguish the AFPD training objective from merely adding
ROI to the action space before any paper-facing claim is made.

## Assumptions

1. Canonical COCO train2017 `fit=9,678` and `calibration=571` are unchanged.
2. COCO val2017 `final=5,000` has already been accessed by prior experiments;
   any evaluation here is explicitly post-hoc final reuse, not a new hidden
   registered final.
3. AFPD v1 architecture, losses, seeds, and hyperparameters are frozen from
   `20260809_coco_frontier_aligned_actionvalue_exploration_v1`.
4. Occupancy Gate v2 uses its frozen 0.5 threshold. CostFree and CostAware are
   secondary ablations and cannot replace the primary method.
5. Existing detector predictions and payload caches are read-only. No detector
   inference and no GPU are required.

## Method matrix

The strict 2x2 attribution matrix is:

| Objective | OQ actions | OQR actions |
|---|---|---|
| Current ActionValue | Current-OQ | Current-OQR |
| AFPD | AFPD-OQ | AFPD-OQR |

Secondary frozen policies applied to AFPD-OQR:

- AFPD-OQR + CostFree Occupancy Gate;
- AFPD-OQR + CostAware Occupancy Gate.

External/reporting references are frozen Raw ActionValue-OQ and ORIC/MORIC
rows from the completed canonical full benchmark. They are never used to tune
the new models.

## Commands

```bash
python -m unittest discover \
  -s paper/experiments/20260810_coco_afpd_occupancy_paper_protocol_v1/tests -v

python paper/experiments/20260810_coco_afpd_occupancy_paper_protocol_v1/scripts/run_experiment.py \
  --stage preflight

python paper/experiments/20260810_coco_afpd_occupancy_paper_protocol_v1/scripts/run_experiment.py \
  --stage all
```

## Project structure

```text
SPEC.md                   frozen scientific contract
PLAN.md                   implementation and execution checkpoints
configs/protocol.json     exact populations, methods, seeds, gates
scripts/core.py           pure protocol/gate/attribution contracts
scripts/run_experiment.py resumable CPU orchestration
tests/                    protocol and scheduling tests
runs/                     generated artifacts only
```

## Code style

Use deterministic experiment-local Python and import completed v1/v2 modules
read-only. Keep all role keys namespaced and assert exact image order.

```python
objective_effect_oq = ap("AFPD-OQ") - ap("Current-OQ")
objective_effect_oqr = ap("AFPD-OQR") - ap("Current-OQR")
interaction = objective_effect_oqr - objective_effect_oq
```

## Testing strategy

- Unit tests first for the exact method matrix, action spaces, row counts,
  delta gates, attribution decomposition, capacity safety, and final gate.
- Preflight asserts source hashes, exact role counts, fit/calibration
  disjointness, frozen folds, and no forbidden model/caches.
- Three seeds x five strict OOF folds precede full-fit prediction.
- Calibration is evaluated once for every frozen row and cannot select a new
  primary method.
- Post-hoc val2017 and bootstrap execute only if the frozen primary calibration
  gate passes.

## Boundaries

- Always: preserve v1/v2 sources; report negative results; distinguish OOF,
  calibration, and post-hoc val2017 evidence; retain exact byte constraints.
- Ask first: new detector inference, new dataset split, or new hyperparameter.
- Never: tune on calibration/val2017; relabel reused val2017 as hidden final;
  promote Occupancy Gate over AFPD based on this evaluation; modify completed
  source experiments.

## Success criteria

1. The 2x2 OOF matrix has exactly 96 unique rows and all Fixed-byte capacities
   are respected.
2. AFPD objective effects are reported separately in OQ and OQR, together with
   the ROI main effect and the objective x ROI interaction.
3. Full-fit checkpoints use all 9,678 fit images, three frozen seeds, and never
   train on calibration/final.
4. Calibration produces 36 frozen ensemble rows: six methods x two modes x
   three budgets.
5. Primary gate: AFPD-OQR versus Current-OQR improves at least two Fixed-byte
   budgets by >=0.001 AP, the remaining budget is >=-0.001, and every
   Fixed-call point is >=-0.001. AFPD-OQR versus Current-OQ is also reported.
6. Occupancy v2 is reported as an ablation regardless of sign and never changes
   the primary gate.
7. If the primary calibration gate passes, post-hoc val2017 and 1,000-replicate
   paired bootstrap run automatically; otherwise final remains untouched by
   this new pipeline.

## Open questions

- Does AFPD improve over a same-action-space Current-OQR baseline?
- Is its benefit specific to the ROI action interaction?
- Does Occupancy Gate's small OOF gain survive full-fit calibration?

