# Spec: COCO DualHead Fixed-Byte Registered Evaluation v1

## Objective

Run a new paper-level COCO evaluation of the frozen DualHead value/cost
separation against the strongest ORIC/MORIC controls. The primary system
question is whether DualHead improves Official AP50-95 under identical Original
upload-byte capacity at 10%, 25%, and 40%.

## Frozen Data Contract

- Training reuses only the 5,000 development images, contextual rewards, and
  detector caches from `20260804_coco_budgetset_registered_paper_v1`.
- The new evaluation roles are 1,500 audit and 3,000 registered-final images.
- Both roles are disjoint from the canonical old 12K and all 9,500 images from
  the preceding registered experiment.
- Audit selects the strongest external comparator at each work point and makes
  the sole go/no-go decision.
- Registered-final inference is forbidden until the audit gate passes. Final is
  accessed once and cannot tune a model, epoch, comparator, or threshold.

## Methods

- External controls: ORIC-AP50, MORIC-AP50, ORIC-AP5095, MORIC-AP5095.
- Primary candidate: frozen DualHead from the completed development OOF
  diagnostic.
- Registered ablations: Current-BudgetSet and SetValue. Ablations cannot affect
  the audit gate or comparator choice.

DualHead keeps direct payload and budget inputs out of the deployed Value head.
The Utility head alone receives cost supervision. Fixed-call ranks by Value;
fixed-byte passes Value plus exact Original bytes to CostEligible, applying cost
once.

## Detector and Payload Contract

- Edge: official YOLO26n; Cloud: official YOLO26x; both use the hashes frozen in
  the preceding registered experiment.
- Original images, imgsz=640, conf=0.001, IoU=0.7, max_det=300, rect=True,
  end2end=False.
- Upload cost is exact source JPEG bytes. Work points are 10%, 25%, and 40%.
- Fixed-byte is primary; fixed-call is secondary and cannot open final.

## Gates

Audit fixed-byte point-estimate gate:

- DualHead gains at least +0.001 AP at two of three budgets against the strongest
  external method frozen independently at that budget;
- the remaining budget loses no more than 0.001 AP;
- every selection respects exact capacity and every isolation audit passes.

Registered-final fixed-byte bootstrap gate:

- 1,000 deterministic paired image-bootstrap replicates;
- 95% CI lower bound is above zero at at least two budgets;
- every remaining lower bound is at least -0.001;
- every DualHead and comparator selection respects the registered capacity.

## Commands

```bash
python -m unittest discover \
  -s paper/experiments/20260804_coco_dualhead_fixedbyte_registered_v1/tests -v
python paper/experiments/20260804_coco_dualhead_fixedbyte_registered_v1/scripts/run_paper.py --stage prepare
python paper/experiments/20260804_coco_dualhead_fixedbyte_registered_v1/scripts/auto_pipeline.py
```

## Project Structure

```text
configs/protocol.json      immutable paths, models, roles, methods, and gates
scripts/paper_core.py      pure split, gate, bootstrap, and isolation contracts
scripts/run_paper.py       prepare/train/audit/open-final/final stages
scripts/infer_original.py  resumable gated Original inference
scripts/auto_pipeline.py   background orchestration
tests/                     unit and stage-boundary tests
runs/                      new artifacts only
notes/results_zh.md        generated registered interpretation
```

## Code Style

Use deterministic typed Python and additive experiment-local files. Reuse
completed primitives read-only and hash every source artifact.

## Testing Strategy

- Unit tests first for disjoint roles, fixed-byte audit gate, comparator freeze,
  final gate, capacity checks, and one-time access.
- AST/source tests ensure training never reads new audit/final labels and final
  inference remains behind audit.
- Two-image Edge/Cloud smoke validates category IDs, paths, hashes, and resume.

## Boundaries

- Always: report all registered work points and failed gates; preserve source
  hashes; write only inside this experiment.
- Ask first: change data roles, methods, detectors, budgets, gate thresholds,
  bootstrap count, or metric priority after prepare.
- Never: tune on audit/final, infer registered final before audit passes, reuse a
  consumed audit as final evidence, or call this COCO test-dev.

## Success Criteria

- Exactly 4,500 new, mutually disjoint evaluation images.
- Frozen training produces all seven registered methods without new reward
  construction.
- Audit produces 7 x 2 x 3 point estimates and deterministic comparator/gate
  artifacts.
- Final stays unopened after a failed audit; after a passed audit it runs once,
  completes 1,000 paired bootstraps, and reports the registered gate honestly.

## Open Questions

None. The user approved the fixed-byte-primary registered route.
