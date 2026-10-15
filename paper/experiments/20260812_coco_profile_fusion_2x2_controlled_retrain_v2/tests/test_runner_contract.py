from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RunnerContractTest(unittest.TestCase):
    def test_runner_never_reuses_an_anchor_score(self):
        source = (ROOT / "scripts/run_experiment.py").read_text(encoding="utf-8")
        self.assertNotIn('if PROTOCOL["cells"][method]["reused_anchor"]', source)
        self.assertNotIn("anchor reproduction failed", source)

    def test_rewards_are_generated_for_all_four_factorial_actions(self):
        source = (ROOT / "scripts/build_rewards.py").read_text(encoding="utf-8")
        self.assertIn("factorial_reward_actions", source)
        self.assertIn("contexts.json", source)

    def test_pipeline_does_not_wait_for_or_modify_active_experiments(self):
        source = (ROOT / "scripts/run_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("wait_for_active_experiments", source)
        self.assertNotIn("build_actions.py", source)


if __name__ == "__main__":
    unittest.main()
