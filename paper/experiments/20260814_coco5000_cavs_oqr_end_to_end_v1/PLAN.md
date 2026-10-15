# Plan

1. Add tests for OQR candidate bytes, ROI request geometry, server coordinate
   restoration, MAUR fallback/fusion, selected-action union, and resumable JSONL.
2. Extend the isolated timing server to map ROI detections back to the full image.
3. Implement a staged, resumable full-set runner reusing the deployed policy,
   exact scheduler, HTTP telemetry, and standard COCO evaluator.
4. Run focused tests and a 10-image OQR smoke through a separate cloud port.
5. Launch the 5,000-image run and preserve progress/status for later inspection.

