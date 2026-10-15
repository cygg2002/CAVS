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


if __name__ == "__main__":
    unittest.main()
