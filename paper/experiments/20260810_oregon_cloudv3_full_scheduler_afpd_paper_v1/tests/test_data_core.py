import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oregon_cloudv3_data_core", ROOT / "scripts/data_core.py"
)
CORE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CORE)
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())


class DataCoreTest(unittest.TestCase):
    def test_rebuilt_observable_action_features_have_no_old_roi_values(self):
        features = list(range(49))
        output = CORE.rebuild_observable_action_features(
            features,
            original_bytes=1000,
            q90_bytes=250,
            roi_payloads=[100, 200],
            roi_areas=[0.2, 0.4],
        )
        self.assertEqual(len(output), 54)
        self.assertEqual(output[-5:], [0.25, 0.1, 0.2, 1.0, 1.0])

    def test_proxy_best_is_deterministic_and_prefers_lower_bytes_on_tie(self):
        qualities = {"original": 0.7, "q90": 0.7, "current@160": 0.6}
        payloads = {"original": 1000, "q90": 200, "current@160": 100}
        self.assertEqual(CORE.proxy_best_action(qualities, payloads), "q90")

    def test_fold_contained_contexts_never_cross_membership(self):
        folds = [[1, 2, 3], [4, 5, 6]]
        contexts = CORE.fold_contained_contexts(folds, context_size=2, seed=9)
        membership = {value: index for index, fold in enumerate(folds) for value in fold}
        self.assertEqual(set(contexts), set(membership))
        for target, values in contexts.items():
            self.assertNotIn(target, values)
            self.assertTrue(all(membership[value] == membership[target] for value in values))

    def test_reward_plan_covers_all_cloud_actions_but_never_final(self):
        plan = CORE.reward_plan(PROTOCOL)
        self.assertEqual(set(plan), {"fit", "calibration"})
        self.assertEqual(plan["fit"], 9923 * 14)
        self.assertEqual(plan["calibration"], 4961 * 14)


if __name__ == "__main__":
    unittest.main()
