# Plan

1. Verify the source protocol, training completion, three score files, and model hash.
2. Load and ensemble the three frozen Current@640+MAUR action-value scores.
3. Run the exact-capacity Fixed-byte scheduler at 10%, 25%, and 40%.
4. Evaluate the frozen Oregon final predictions with MAUR fusion.
5. Write isolated CSV/JSON/Chinese-summary artifacts and verify contracts.

