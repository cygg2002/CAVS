# Plan

1. Verify model hashes, policy package, dataset, SSH tunnel, and remote GPU state.
2. Add unit tests for deterministic sampling, byte accounting, timing summaries,
   O/Q scheduling, failure fallback, and COCO prediction conversion.
3. Implement an isolated instrumented cloud service and batch replay benchmark.
4. Run local tests and a 10-image remote smoke benchmark.
5. Run the frozen formal subset, generate AP/latency/byte tables, and explain the
   offline-theory versus real-system gap without claiming RK3588 performance.

