# RKNN-FP16 Deployment-Aligned Real HTTP Run

This package evaluates the newly retrained RKNN-FP16 CAVS-OQR policy. It is
separate from the old paper-policy deployment and does not reuse old candidates,
plans, requests, or results.

## Protocol Scope

This is the paper-aligned two-pass Fixed-byte experiment. Pass 1 runs RKNN on
all 5,000 images and freezes the 10%/25%/40% global plans. Pass 2 performs the
selected uploads over real HTTP and runs the cloud detector. It is a real
hardware-in-the-loop execution of the paper scheduler, but it is not a
zero-lookahead causal online scheduler.

## Mac Mini Cloud

The service is started by the experiment owner and must report:

- URL: `http://192.168.40.78:8767/infer`
- Cloud SHA-256: `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`
- Authentication: required

Keep the provided `CAVS_TOKEN` only in the shell environment.

## Orange Pi Preflight

After mounting the USB drive:

```bash
cd /mnt/usb/rk3588/cavs_oqr_deployment_aligned_fp16
export CAVS_TOKEN='<token-provided-with-this-run>'
curl http://192.168.40.78:8767/health

bash run_board.sh --model fp16 --stage all --limit 50 \
  --run-id smoke_deploy_aligned_fp16_$(date +%Y%m%d_%H%M%S) \
  --cloud-url http://192.168.40.78:8767/infer
```

The 50-image result verifies the chain only and is not paper-comparable.

## Formal 5,000 Images

Use a new run ID after smoke succeeds:

```bash
bash run_board.sh --model fp16 --stage all \
  --run-id formal_deploy_aligned_fp16_$(date +%Y%m%d_%H%M%S) \
  --cloud-url http://192.168.40.78:8767/infer
```

The process is resumable: after an interruption, repeat the exact command with
the same run ID. A different model, policy, protocol, limit, or budget under an
existing run ID is rejected. Any missing or mismatched cloud response also
stops the run instead of silently producing an Edge-fallback result. Final
results are written to:

```text
runs/<run-id>/fp16/summary.json
```

The formal summary must report `paper_comparable_fixed_byte=true`, 5,000 images,
the frozen policy SHA-256, and successful requests equal to required requests.
