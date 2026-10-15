# INT8-heads CAVS-OQR causal online HIL

## Objective

Measure the deployment effect of replacing the completed RK3588 FP16 edge
model with `yolo26n-rk3588-int8-heads.rknn`, without changing the cloud model,
COCO sequence, actions, byte definition, binary transport, or zero-lookahead
execution semantics.

## Frozen comparison

- Method: only CAVS-OQR (`Edge`, `Original`, `Q90`, `Current@640 + MAUR`).
- Edge policy: the three-seed CAVS policy trained from INT8-heads candidates;
  it is not the FP16 policy and no learner is retrained in this experiment.
- Budgets: JPEG-payload 10%, 25%, and 40% with one-frame projected
  primal-dual control frozen from the 571-image calibration role.
- Final sequence: 5,000 images, immediate upload before reading the next
  frame, zero look-ahead, no global plan and no labels during execution.
- Cloud: hash-pinned `yolo26x.pt` (`9fdd44…7003e92`), binary wire v2.
- Edge: hash-pinned INT8-heads RKNN (`9da11e…a2ad5`).

## Acceptance gates

For each budget: 5,000 committed frames in canonical order, no fallback
responses, zero prefix-budget violation, and HTTP-body/JPEG-payload ratio at
most 1.02.  The run contract must record the new INT8-specific protocol and
both model hashes.

## Scope

This is a quantization robustness ablation, not a replacement of the 46-run
FP16 system matrix.  External baselines remain FP16 unless a separate,
fully symmetric INT8 matrix is explicitly requested.
