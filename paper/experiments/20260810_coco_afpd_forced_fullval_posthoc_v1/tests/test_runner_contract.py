from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RunnerContractTests(unittest.TestCase):
    def test_runner_retrains_every_stage_and_never_copies_checkpoints(self):
        text = (ROOT / "scripts/run_experiment.py").read_text(encoding="utf-8")
        for call in (
            "SOURCE.run_action_oof_training()",
            "SOURCE.run_occupancy_oof_training()",
            "SOURCE.run_calibration_training()",
        ):
            self.assertIn(call, text)
        self.assertNotIn("copy2(", text)
        self.assertNotIn("os.link(", text)

    def test_runner_discloses_unconditional_posthoc_full_val(self):
        text = (ROOT / "scripts/run_experiment.py").read_text(encoding="utf-8")
        self.assertIn('"confirmatory": False', text)
        self.assertIn('"final_is_hidden": False', text)
        self.assertIn('"full_val_unconditional": True', text)

    def test_runner_includes_oric_bootstrap(self):
        text = (ROOT / "scripts/run_experiment.py").read_text(encoding="utf-8")
        self.assertIn("AFPD_vs_ORIC", text)
        self.assertIn("ORIC-MLP|SharedAction-NoROI", text)

    def test_preflight_hashes_the_actual_v1_and_v2_training_modules(self):
        text = (ROOT / "scripts/run_experiment.py").read_text(encoding="utf-8")
        self.assertNotIn('SOURCE_ROOT / "scripts/training.py"', text)
        self.assertIn(
            'SOURCE.SOURCES["afpd_v1"] / "scripts/training.py"', text
        )
        self.assertIn(
            'SOURCE.SOURCES["occupancy_v2"] / "scripts/training.py"', text
        )


if __name__ == "__main__":
    unittest.main()
