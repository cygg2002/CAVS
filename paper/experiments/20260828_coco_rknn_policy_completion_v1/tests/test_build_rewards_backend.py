from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "rknn_build_rewards_backend", SCRIPTS / "build_rewards_backend.py"
)
assert SPEC and SPEC.loader
REWARDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REWARDS)


class BuildRewardsBackendTests(unittest.TestCase):
    def test_reward_actions_match_deployment_action_space(self) -> None:
        self.assertEqual(
            REWARDS.reward_actions(),
            ("original", "q90", "current@640"),
        )

    def test_role_root_is_backend_isolated(self) -> None:
        self.assertEqual(
            REWARDS.role_root(ROOT, "fp16", "fit"),
            ROOT / "runs/backends/fp16/common/coco/fit",
        )
        with self.assertRaises(ValueError):
            REWARDS.role_root(ROOT, "fp16", "final-reward")


if __name__ == "__main__":
    unittest.main()
