from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


HOST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST))

from summarize_online_run import summarize_frames


class SummarizeOnlineRunTest(unittest.TestCase):
    def test_reports_prefix_capacity_actions_failures_and_latency(self) -> None:
        rows = [
            {
                "frame_start_ns": 1_000_000,
                "decision_ns": 2_000_000,
                "frame_done_ns": 11_000_000,
                "decision": {
                    "action": "edge", "payload_bytes": 0, "original_bytes": 100,
                    "cumulative_original_bytes": 100, "spent_bytes": 0,
                    "credit_floor_bytes": 10,
                },
                "execution_status": "not_required",
                "cloud": {"request_body_bytes": 0, "response_body_bytes": 0},
                "timings_ms": {"edge_inference": 4.0, "candidate": 1.0},
            },
            {
                "frame_start_ns": 20_000_000,
                "decision_ns": 23_000_000,
                "frame_done_ns": 40_000_000,
                "decision": {
                    "action": "q90", "payload_bytes": 20, "original_bytes": 100,
                    "cumulative_original_bytes": 200, "spent_bytes": 20,
                    "credit_floor_bytes": 0,
                },
                "execution_status": "fallback",
                "cloud": {"request_body_bytes": 80, "response_body_bytes": 10},
                "timings_ms": {
                    "edge_inference": 5.0, "candidate": 2.0,
                    "cloud_roundtrip": 10.0,
                },
            },
        ]

        summary = summarize_frames(rows, budget_numerator=1, budget_denominator=10)

        self.assertEqual(summary["frames"], 2)
        self.assertEqual(summary["action_counts"], {"edge": 1, "q90": 1})
        self.assertEqual(summary["uploads"], 1)
        self.assertEqual(summary["cloud_failures"], 1)
        self.assertEqual(summary["selected_payload_bytes"], 20)
        self.assertEqual(summary["capacity_floor_bytes"], 20)
        self.assertEqual(summary["capacity_utilization"], 1.0)
        self.assertEqual(summary["maximum_prefix_violation_bytes"], 0)
        self.assertEqual(summary["wire_request_body_bytes"], 80)
        self.assertEqual(summary["e2e_ms"]["mean"], 15.0)
        self.assertEqual(summary["timing_ms"]["edge_inference"]["mean"], 4.5)

    def test_rejects_a_prefix_budget_violation(self) -> None:
        row = {
            "frame_start_ns": 1, "decision_ns": 2, "frame_done_ns": 3,
            "decision": {
                "action": "original", "payload_bytes": 11, "original_bytes": 100,
                "cumulative_original_bytes": 100, "spent_bytes": 11,
            },
            "execution_status": "ok", "cloud": {}, "timings_ms": {},
        }
        with self.assertRaisesRegex(RuntimeError, "prefix byte budget"):
            summarize_frames([row], budget_numerator=1, budget_denominator=10)


if __name__ == "__main__":
    unittest.main()
