from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RunnerContractTest(unittest.TestCase):
    def test_runner_uses_frozen_hybrid_composition_and_complete_matrix(self):
        text = (ROOT / "scripts/run_experiment.py").read_text(encoding="utf-8")
        self.assertIn("CORE.compose_hybrid_scores", text)
        self.assertIn("expected_rows = 24", text)
        self.assertIn("expected_comparisons = 12", text)

    def test_runner_discloses_posthoc_final_reuse(self):
        text = (ROOT / "scripts/run_experiment.py").read_text(encoding="utf-8")
        self.assertIn('"final_is_hidden": False', text)
        self.assertIn('"final_used_for_selection": False', text)
        self.assertIn('"confirmatory": False', text)


if __name__ == "__main__":
    unittest.main()
