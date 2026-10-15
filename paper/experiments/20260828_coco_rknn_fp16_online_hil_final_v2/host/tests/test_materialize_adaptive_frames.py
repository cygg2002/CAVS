from __future__ import annotations

from pathlib import Path
import sys
import unittest


HOST = Path(__file__).resolve().parents[1]
BOARD_SCRIPTS = HOST.parent / "board_bundle" / "scripts"
sys.path.insert(0, str(HOST))
sys.path.insert(0, str(BOARD_SCRIPTS))

from materialize_adaptive_frames import materialize_frames


class MaterializeAdaptiveFramesTest(unittest.TestCase):
    def test_only_changed_cloud_actions_are_executed(self) -> None:
        source = [
            {
                "position": 0, "image_id": 1, "file_name": "1.jpg",
                "status": "committed",
                "decision": {"action": "q90"},
                "edge_boxes": [{"xyxy": [0.0, 0.0, 2.0, 2.0], "conf": 0.5, "cls": 0}],
                "resolved_boxes": [{"xyxy": [0.0, 0.0, 3.0, 3.0], "conf": 0.6, "cls": 0}],
            },
            {
                "position": 1, "image_id": 2, "file_name": "2.jpg",
                "status": "committed",
                "decision": {"action": "q90"},
                "edge_boxes": [{"xyxy": [1.0, 1.0, 2.0, 2.0], "conf": 0.4, "cls": 1}],
                "resolved_boxes": [{"xyxy": [1.0, 1.0, 3.0, 3.0], "conf": 0.7, "cls": 1}],
            },
            {
                "position": 2, "image_id": 3, "file_name": "3.jpg",
                "status": "committed",
                "decision": {"action": "edge"},
                "edge_boxes": [],
                "resolved_boxes": [],
            },
        ]
        calls = []

        def execute_cloud(row, action):
            calls.append((row["image_id"], action))
            return [{"xyxy": [2.0, 2.0, 4.0, 4.0], "conf": 0.8, "cls": 2}]

        result, diagnostics = materialize_frames(
            source,
            adaptive_selections={"1": "q90", "2": "edge", "3": "original"},
            execute_cloud=execute_cloud,
        )
        self.assertEqual(calls, [(3, "original")])
        self.assertEqual(result[0]["resolved_boxes"], source[0]["resolved_boxes"])
        self.assertEqual(result[1]["resolved_boxes"], source[1]["edge_boxes"])
        self.assertEqual(result[2]["resolved_boxes"][0]["cls"], 2)
        self.assertEqual(diagnostics["reused_frames"], 1)
        self.assertEqual(diagnostics["edge_replacements"], 1)
        self.assertEqual(diagnostics["cloud_executions"], 1)

    def test_rejects_missing_or_extra_selections(self) -> None:
        row = {
            "position": 0, "image_id": 1, "file_name": "1.jpg",
            "status": "committed", "decision": {"action": "edge"},
            "edge_boxes": [], "resolved_boxes": [],
        }
        for selections in ({}, {"1": "edge", "2": "edge"}):
            with self.assertRaises(ValueError):
                materialize_frames(
                    [row],
                    adaptive_selections=selections,
                    execute_cloud=lambda source, action: [],
                )

    def test_filters_degenerate_boxes_only_in_the_derived_evaluation_view(self) -> None:
        row = {
            "position": 0, "image_id": 1, "file_name": "1.jpg",
            "status": "committed", "decision": {"action": "q90"},
            "edge_boxes": [],
            "resolved_boxes": [
                {"xyxy": [1.0, 1.0, 1.0, 2.0], "conf": 0.01, "cls": 0},
                {"xyxy": [1.0, 1.0, 2.0, 2.0], "conf": 0.9, "cls": 0},
            ],
        }
        frames, diagnostics = materialize_frames(
            [row],
            adaptive_selections={"1": "q90"},
            execute_cloud=lambda source, action: [],
        )
        self.assertEqual(len(frames[0]["resolved_boxes"]), 1)
        self.assertEqual(diagnostics["degenerate_boxes_filtered"], 1)
        self.assertEqual(len(row["resolved_boxes"]), 2)


if __name__ == "__main__":
    unittest.main()
