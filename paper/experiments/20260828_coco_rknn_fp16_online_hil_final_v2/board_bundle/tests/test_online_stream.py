from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from online_control import CausalByteController
from online_stream import execute_stream


class OnlineStreamTest(unittest.TestCase):
    def test_frame_finishes_before_next_frame_is_observed(self) -> None:
        controller = CausalByteController(
            budget="0.4",
            price=0.0,
            median_original_bytes=100,
            action_ids=("edge", "original"),
        )
        events = []
        committed = []

        def observe(item):
            events.append(("open", item["image_id"]))
            return {
                "frame_id": item["image_id"],
                "payload_bytes": {"edge": 0, "original": 100},
                "scores": {"edge": 0.0, "original": 1.0},
            }

        def execute(item, observation, decision):
            events.append(("request", item["image_id"]))
            events.append(("done", item["image_id"]))
            return {"status": "ok", "resolved_boxes": []}

        execute_stream(
            [{"image_id": 1}, {"image_id": 2}],
            controller=controller,
            observe=observe,
            execute=execute,
            commit=committed.append,
        )
        self.assertEqual(
            events,
            [
                ("open", 1),
                ("request", 1),
                ("done", 1),
                ("open", 2),
                ("request", 2),
                ("done", 2),
            ],
        )
        self.assertEqual([row["position"] for row in committed], [0, 1])
        self.assertLessEqual(committed[0]["frame_done_ns"], committed[1]["frame_start_ns"])

    def test_resume_starts_after_committed_prefix(self) -> None:
        controller = CausalByteController(
            budget="0.4",
            price=0.0,
            median_original_bytes=100,
            action_ids=("edge", "original"),
        )
        opened = []

        def observe(item):
            opened.append(item["image_id"])
            return {
                "frame_id": item["image_id"],
                "payload_bytes": {"edge": 0, "original": 100},
                "scores": {"edge": 0.0, "original": 1.0},
            }

        execute_stream(
            [{"image_id": 1}, {"image_id": 2}, {"image_id": 3}],
            controller=controller,
            observe=observe,
            execute=lambda item, observation, decision: {"status": "ok"},
            commit=lambda row: None,
            start_position=2,
        )
        self.assertEqual(opened, [3])

    def test_observation_cannot_change_frame_identity(self) -> None:
        controller = CausalByteController(
            budget="0.4",
            price=0.0,
            median_original_bytes=100,
            action_ids=("edge", "original"),
        )
        with self.assertRaises(RuntimeError):
            execute_stream(
                [{"image_id": 1}],
                controller=controller,
                observe=lambda item: {
                    "frame_id": 2,
                    "payload_bytes": {"edge": 0, "original": 100},
                    "scores": {"edge": 0.0, "original": 1.0},
                },
                execute=lambda item, observation, decision: {},
                commit=lambda row: None,
            )


if __name__ == "__main__":
    unittest.main()
