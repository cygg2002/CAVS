import importlib.util
from pathlib import Path
import unittest


CORE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "core.py"


def load_core():
    spec = importlib.util.spec_from_file_location("twostage_raw_core", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RawProjectionTests(unittest.TestCase):
    def test_projection_preserves_raw_values_and_excludes_edge(self):
        core = load_core()
        projected = core.raw_projected_score_map(
            {
                1: {"edge": 99.0, "original": 10.0, "q90": 20.0},
                2: {"edge": 99.0, "original": -1.0, "q90": -2.0},
            }
        )

        self.assertEqual(
            projected,
            {
                1: {"q90": 20.0},
                2: {"original": -1.0},
            },
        )

    def test_projection_uses_deterministic_action_order_for_ties(self):
        core = load_core()
        projected = core.raw_projected_score_map(
            {7: {"q90": 0.5, "original": 0.5}},
            action_order=("original", "q90"),
        )

        self.assertEqual(projected, {7: {"original": 0.5}})

    def test_projection_rejects_images_without_cloud_actions(self):
        core = load_core()

        with self.assertRaisesRegex(ValueError, "image 3"):
            core.raw_projected_score_map({3: {"edge": 1.0}})


if __name__ == "__main__":
    unittest.main()
