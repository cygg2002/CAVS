from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scheduler import schedule_fixed_byte


class SchedulerTest(unittest.TestCase):
    def test_plan_stays_within_capacity_and_is_deterministic(self) -> None:
        candidates = [
            {
                "frame_id": 1,
                "scores": {"edge": 0.0, "original": 0.9, "q90": 0.7, "current@640": 0.6},
                "payload_bytes": {"edge": 0, "original": 1000, "q90": 500, "current@640": 200},
            },
            {
                "frame_id": 2,
                "scores": {"edge": 0.0, "original": 0.8, "q90": 0.65, "current@640": 0.1},
                "payload_bytes": {"edge": 0, "original": 1000, "q90": 500, "current@640": 250},
            },
            {
                "frame_id": 3,
                "scores": {"edge": 0.0, "original": 0.2, "q90": 0.15},
                "payload_bytes": {"edge": 0, "original": 1000, "q90": 500},
            },
        ]
        plan = schedule_fixed_byte(candidates, 0.25)
        self.assertEqual(plan["capacity_bytes"], 750.0)
        self.assertLessEqual(plan["request_bytes"], plan["capacity_bytes"])
        self.assertEqual(set(plan["actions"]), {1, 2, 3})

    def test_equal_values_prefer_lower_cost(self) -> None:
        plan = schedule_fixed_byte(
            [{
                "frame_id": 7,
                "scores": {"edge": 0.0, "original": 0.5, "q90": 0.5, "current@640": 0.5},
                "payload_bytes": {"edge": 0, "original": 1000, "q90": 500, "current@640": 200},
            }],
            1.0,
        )
        self.assertEqual(plan["actions"], {7: "current@640"})
        self.assertEqual(plan["request_bytes"], 200)


if __name__ == "__main__":
    unittest.main()

