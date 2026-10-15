from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_pipeline import completion_path, dataset_command  # noqa: E402


class PipelineRunnerTest(unittest.TestCase):
    def test_dataset_command_uses_current_python_and_isolated_runner(self) -> None:
        command = dataset_command("coco", "all")
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(Path(command[1]), ROOT / "scripts/run_dataset.py")
        self.assertEqual(command[-4:], ["--dataset", "coco", "--stage", "all"])

    def test_dataset_command_rejects_unknown_values(self) -> None:
        with self.assertRaises(ValueError):
            dataset_command("unknown", "all")
        with self.assertRaises(ValueError):
            dataset_command("coco", "unknown")

    def test_evaluate_stage_uses_evaluation_completion_directory(self) -> None:
        self.assertEqual(
            completion_path("coco", "evaluate"),
            ROOT / "runs/coco/evaluation/completion.json",
        )


if __name__ == "__main__":
    unittest.main()
