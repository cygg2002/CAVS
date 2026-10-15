# Oregon YOLO26x Cloud-V3 30-Epoch Fine-tune

## Objective

Repeat the actual Cloud-V2 20-epoch training recipe for 30 new epochs, initialized
from the frozen Cloud-V2 `best.pt`, without modifying the Cloud-V2 run.

## Frozen protocol

- Dataset: Oregon same-camera event-disjoint detector train/val.
- Model: O2M-only YOLO26x configuration (`end2end: false`).
- Initialization: Cloud-V2 `best.pt`, SHA-256 pinned in `configs/protocol.json`.
- Training: 30 epochs, image size 640, real batch 8, device 0.
- The optimizer, augmentation, AMP, seed, and validation settings reproduce the
  actual `finetune20_batch8_from_e18_restart2/args.yaml` values.
- This is a fresh fine-tune from stripped Cloud-V2 weights, not optimizer-state
  resume from `epoch19.pt`.

## Boundaries

- Never overwrite or edit the Cloud-V2 source run.
- Write training products only to the new run name.
- Do not use scheduler audit/final data for detector training or checkpoint selection.
- Treat Cloud-V3 as a new detector protocol; existing Cloud-V1/V2 caches are not
  comparable without regeneration.

## Success criteria

1. Preflight validates the source hash, model/data paths, O2M model flag, and the
   absence of the destination run.
2. The background process starts and creates `args.yaml` with the frozen settings.
3. Training enters epoch 1/30 with real batch 8.

