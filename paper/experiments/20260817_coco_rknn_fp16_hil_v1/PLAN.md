# Implementation Plan

## Phase 1: Freeze Contracts

- [x] Copy only the proven two-pass client and cloud server dependencies.
- [x] Add failing tests for cross-backend and cross-model policy use.

## Phase 2: Build FP16-Only Bundle

- [x] Pin the new deployment-aligned policy and protocol hashes.
- [x] Restrict the launcher and verifier to FP16.
- [x] Run the full local bundle test suite and real hash verification.

## Phase 3: Stage and Serve

- [x] Synchronize the clean bundle to the USB without old runs.
- [x] Start an authenticated Mac LAN cloud service.
- [x] Verify `/health` locally and provide board smoke/formal commands.

## Risks

- A stale run ID can silently reuse old candidates. Mitigation: isolated bundle
  plus unique run IDs.
- A policy can be paired with the wrong RKNN backend. Mitigation: manifest
  backend and RKNN SHA checks in both preflight and board-client validation.
- The USB is removed from the Mac for board use. Mitigation: run the cloud
  service from the local experiment mirror, not from the USB mount.
- LAN HTTP is unencrypted. Mitigation: bind only to `192.168.40.78`, require a
  random bearer token, and stop the service after the temporary experiment.
