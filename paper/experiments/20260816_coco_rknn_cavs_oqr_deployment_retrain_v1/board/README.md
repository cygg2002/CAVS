# RK3588 Candidate Collection

This directory contains the board-only collection stage. It must run on the
Orange Pi with the RKNN runtime; learner training remains on the host.

## Host preparation

On the training host, where the frozen canonical cache and COCO `train2017`
images are available:

```bash
python scripts/prepare_board_inputs.py \
  --output /path/to/rk3588/deployment_aligned_inputs \
  --copy-images --copy-roles fit calibration
```

Copy `deployment_aligned_inputs/manifests` and the corresponding
`images/train2017` or `images/val2017` directories to the board. The existing
`val2017` bundle can be used for the final role if its image hash manifest
matches the canonical final samples.

## Board collection

Copy `collect_candidates.py` into the board bundle's `scripts` directory. Run
one backend at a time, and run all three roles for that backend before moving
to the other backend:

```bash
python scripts/collect_candidates.py \
  --model fp16 --role fit \
  --manifest deployment_aligned_inputs/manifests/fit.json \
  --images-root deployment_aligned_inputs/images/train2017 \
  --output-dir runs/deployment_aligned/fp16/fit

python scripts/collect_candidates.py \
  --model fp16 --role calibration \
  --manifest deployment_aligned_inputs/manifests/calibration.json \
  --images-root deployment_aligned_inputs/images/train2017 \
  --output-dir runs/deployment_aligned/fp16/calibration

python scripts/collect_candidates.py \
  --model fp16 --role final \
  --manifest deployment_aligned_inputs/manifests/final.json \
  --images-root deployment_aligned_inputs/images/val2017 \
  --output-dir runs/deployment_aligned/fp16/final
```

Repeat the three commands with `--model int8-heads` and matching output
directories. `candidates.jsonl` is resumable and fails closed on duplicate
IDs, role drift, model-hash drift, or missing images. It contains no old
policy scores.
