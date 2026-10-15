from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RunnerContractTests(unittest.TestCase):
    def test_runner_waits_for_parent_before_bootstrap(self):
        text = (ROOT / "scripts/run_experiment.py").read_text(encoding="utf-8")
        self.assertIn("wait_for_parent", text)
        self.assertIn('PROTOCOL["parent_completion"]', text)
        self.assertLess(text.index("wait_for_parent"), text.index("mp.get_context"))

    def test_runner_has_explicit_concurrent_override(self):
        text = (ROOT / "scripts/run_experiment.py").read_text(encoding="utf-8")
        self.assertIn("--allow-concurrent-parent", text)
        self.assertIn("allow_concurrent_parent", text)
        self.assertIn('"concurrent_with_parent_bootstrap"', text)

    def test_runner_is_read_only_with_respect_to_parent(self):
        text = (ROOT / "scripts/run_experiment.py").read_text(encoding="utf-8")
        self.assertNotIn("run_training(", text)
        self.assertNotIn("run_final_prediction(", text)
        self.assertNotIn("run_final_evaluation(", text)
        self.assertIn('"confirmatory": False', text)
        self.assertIn('"final_is_hidden": False', text)

    def test_wait_script_gates_on_parent_completion_before_python(self):
        text = (ROOT / "scripts/wait_then_run.sh").read_text(encoding="utf-8")
        self.assertIn("runs/bootstrap/completion.json", text)
        self.assertIn("sleep 30", text)
        self.assertLess(text.index("while !"), text.index('exec "$PYTHON_BIN"'))


if __name__ == "__main__":
    unittest.main()
