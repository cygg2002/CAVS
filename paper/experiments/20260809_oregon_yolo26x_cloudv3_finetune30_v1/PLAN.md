# Execution Plan

1. Validate the frozen protocol and source artifacts with `--preflight-only`.
2. Launch the isolated training process with `nohup` on GPU 0.
3. Verify PID liveness, generated `args.yaml`, source-weight transfer, and entry
   into epoch 1/30.
4. Leave the 30-epoch process running unattended; do not change its protocol.

