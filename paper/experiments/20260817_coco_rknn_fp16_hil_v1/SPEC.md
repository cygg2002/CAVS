# Spec: COCO RKNN-FP16 Deployment-Aligned Hardware-in-the-Loop

## Objective

Run the retrained RKNN-FP16 CAVS-OQR policy on Orange Pi 5 Ultra and execute
the selected Original, Q90, and Current@640 actions against the Mac mini over
the local LAN. The experiment must preserve the paper fixed-byte 10%, 25%, and
40% protocol while keeping all earlier bundles and results unchanged.

## Frozen Assets

- Dataset: COCO val2017, 5,000 images, native JPEG denominator 814,705,164 bytes.
- Edge: `yolo26n-rk3588-fp16.rknn`, SHA-256
  `0915a99235c3be5723fcfeda45b06775ecccc2908cc5b490f093f23bbdede12d`.
- Policy: `cavs_current_maur_rknn_fp16_v1`, three seeds, manifest SHA-256
  `1f057ff5f5417cdeabd7c35706b5c5d2493039655fdf01381e602c10f57111c6`.
- Cloud: `yolo26x.pt`, SHA-256
  `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`.
- Actions: Edge, Original, Q90, Current@640 with MAUR fusion.

## Commands

Mac server:

```bash
export CAVS_TOKEN='<runtime-only-token>'
export CLOUD_PYTHON='/Users/cygg_macmini/work/Codex/ultralytics-main/.venv/bin/python'
bash mac_server/run_cloud.sh \
  --model /Users/cygg_macmini/work/Codex/ultralytics-main/yolo26x.pt \
  --host 192.168.40.78 --port 8767 --device mps
```

Orange Pi smoke and formal commands are documented in `README.md` after the
bundle is synchronized to `/mnt/usb/rk3588/cavs_oqr_deployment_aligned_fp16`.

## Project Structure

- `board_bundle/`: immutable FP16-only board client copied to the USB drive.
- `mac_server/`: frozen cloud HTTP server implementation.
- `tests/`: contract tests for backend/model binding and server protocol.
- `runs/`: Mac server logs only; board results remain on the USB drive.

## Code Style

Use the existing Python and shell style. Keep validation explicit and fail
closed when policy, RKNN model, cloud model, dataset size, or byte denominator
does not match the frozen contract.

## Testing Strategy

- Unit tests must reject a policy whose backend or RKNN SHA-256 differs from
  the selected board model.
- Existing scheduler, codec, MAUR, request validation, and NumPy policy tests
  must continue to pass.
- `verify_bundle.py` must pass against the real sibling dataset and model.
- The Mac `/health` response must match the frozen cloud hash and inference
  protocol before the board uploads any image.

## Boundaries

- Always: use a fresh run ID; keep the three policy files and manifest together;
  require a runtime bearer token; bind only to the current LAN address.
- Ask first: changing model files, dataset, action semantics, budgets, or cloud
  inference flags.
- Never: overwrite the old `cavs_oqr_edge_cloud` bundle, reuse its candidates or
  requests, expose the service to the public Internet, or call cached AP a real
  HTTP result.

## Success Criteria

1. The board bundle accepts only FP16 and verifies both policy and RKNN hashes.
2. All unit and bundle-contract tests pass.
3. The Mac service reports the exact cloud model hash and requires authentication.
4. A 50-image smoke run completes with no failed request.
5. The same package can run all 5,000 images at 10%, 25%, and 40% fixed-byte
   budgets without changing code or configuration.

## Open Questions

None. The user explicitly selected the Mac as the temporary cloud endpoint on
the same trusted LAN as the Orange Pi.
