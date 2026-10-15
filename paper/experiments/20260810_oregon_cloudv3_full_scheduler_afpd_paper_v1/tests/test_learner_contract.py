import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oregon_cloudv3_learner_core", ROOT / "scripts/learner_core.py"
)
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())


class LearnerContractTest(unittest.TestCase):
    def test_complete_evaluation_matrix_is_frozen(self):
        matrix = CORE.evaluation_matrix(PROTOCOL)
        # 16 scalar methods x 5 policies plus 13 joint methods, at 6 work points.
        self.assertEqual(len(matrix), (16 * 5 + 13) * 2 * 3)
        self.assertEqual(len({CORE.result_key(row) for row in matrix}), len(matrix))

    def test_action_score_map_uses_record_action_ids_not_column_names(self):
        records = [
            {
                "image_id": 7,
                "actions": [
                    {"action_id": "edge"},
                    {"action_id": "original"},
                    {"action_id": "historical_cost@320"},
                ],
            }
        ]
        values = np.asarray([[0.0, 0.4, 0.9, -100.0]])
        self.assertEqual(
            CORE.action_score_map(records, values),
            {7: {"original": 0.4, "historical_cost@320": 0.9}},
        )

    def test_occupancy_gate_is_action_specific_for_r12(self):
        scores = {
            3: {"original": 0.4, "current@160": 0.8, "moderate@320": 0.7}
        }
        probability = {
            3: {"current@160": 0.2, "moderate@320": 0.9}
        }
        gated = CORE.gate_roi_scores(scores, probability, threshold=0.5)
        self.assertLess(gated[3]["current@160"], -1e8)
        self.assertEqual(gated[3]["moderate@320"], 0.7)
        self.assertEqual(gated[3]["original"], 0.4)

    def test_final_never_appears_in_training_jobs(self):
        for row in CORE.training_role_matrix(PROTOCOL):
            self.assertNotEqual(row["prediction_role"], "final" if row["fold"] != "full" else "fit")
            if row["fold"] == "full":
                self.assertEqual(row["training_role"], "fit")
                self.assertEqual(row["prediction_roles"], ["calibration", "final"])


if __name__ == "__main__":
    unittest.main()
