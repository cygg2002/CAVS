import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "voc_training_under_test", ROOT / "scripts/train_learners.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TrainingJobTest(unittest.TestCase):
    def test_fold_job_excludes_holdout_and_full_job_uses_every_sample(self):
        identities = ["a", "b", "c", "d", "e", "f"]
        folds = [["a", "d"], ["b", "e"], ["c", "f"]]
        train, holdout = MODULE.fold_train_holdout_indices(identities, folds, 1)
        self.assertEqual([identities[index] for index in holdout], ["b", "e"])
        self.assertEqual(set(identities[index] for index in train), {"a", "c", "d", "f"})
        train, holdout = MODULE.fold_train_holdout_indices(identities, folds, "full")
        self.assertEqual(train, list(range(6)))
        self.assertEqual(holdout, [])


if __name__ == "__main__":
    unittest.main()
