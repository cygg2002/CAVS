import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "voc_rewards_under_test", ROOT / "scripts/build_rewards.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _Metric:
    def evaluate(self, predictions, image_ids):
        value = sum(float(row["score"]) for row in predictions) / len(image_ids)
        return {"map50_95": value, "map50": value * 2.0}


class RewardTest(unittest.TestCase):
    def test_contextual_delta_is_scaled_by_context_population(self):
        result = MODULE.contextual_reward(
            _Metric(),
            [1, 2],
            [{"image_id": 1, "score": 0.0}, {"image_id": 2, "score": 0.0}],
            [{"image_id": 1, "score": 1.0}, {"image_id": 2, "score": 0.0}],
        )
        self.assertAlmostEqual(result["ap50_95"], 1.0)
        self.assertAlmostEqual(result["ap50"], 2.0)

    def test_reward_views_are_independent_and_oqr_contains_roi(self):
        rows = [{
            "image_id": 7,
            "rewards": {
                "original": {"ap50_95": 1.0, "ap50": 2.0},
                "q90": {"ap50_95": 0.8, "ap50": 1.8},
                "current@official": {"ap50_95": 1.2, "ap50": 2.2},
            },
        }]
        views, metric = MODULE.build_reward_views(rows)
        self.assertEqual(views["O"], {7: {"original": 1.0}})
        self.assertEqual(views["OQ"], {7: {"original": 1.0, "q90": 0.8}})
        self.assertEqual(
            views["OQR"],
            {7: {"original": 1.0, "q90": 0.8, "current@official": 1.2}},
        )
        self.assertEqual(metric[7]["current@official"]["ap50"], 2.2)


if __name__ == "__main__":
    unittest.main()
