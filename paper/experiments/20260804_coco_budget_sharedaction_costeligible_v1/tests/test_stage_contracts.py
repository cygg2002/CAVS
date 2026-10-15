from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "scripts"))

from experiment_core import (  # noqa: E402
    assert_holdout_allowed,
    validate_protocol,
    validate_split_manifest,
)


class StageContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(
            (EXPERIMENT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_frozen_protocol_is_valid(self) -> None:
        validate_protocol(self.protocol)

    def test_protocol_rejects_holdout_or_val_final_leakage(self) -> None:
        changed = dict(self.protocol)
        changed["holdout_used_for_training"] = True
        with self.assertRaises(ValueError):
            validate_protocol(changed)
        changed = dict(self.protocol)
        changed["val2017_final_accessed"] = True
        with self.assertRaises(ValueError):
            validate_protocol(changed)

    def test_split_manifest_rejects_overlap_or_population_drift(self) -> None:
        manifest = {
            "train_image_ids": [1, 2, 3],
            "holdout_image_ids": [4, 5],
            "split_seed": 7,
        }
        validate_split_manifest(manifest, expected_images=5, expected_train=3)
        manifest["holdout_image_ids"] = [3, 4]
        with self.assertRaises(ValueError):
            validate_split_manifest(manifest, expected_images=5, expected_train=3)

    def test_holdout_requires_completed_training_and_one_time_access(self) -> None:
        assert_holdout_allowed({"complete": True, "holdout_accessed": False}, False)
        with self.assertRaises(RuntimeError):
            assert_holdout_allowed({"complete": False}, False)
        with self.assertRaises(RuntimeError):
            assert_holdout_allowed({"complete": True}, True)

    def test_runner_does_not_reference_consumed_val2017_final(self) -> None:
        runner = (EXPERIMENT / "scripts/run_experiment.py")
        if not runner.exists():
            self.fail("run_experiment.py is not implemented")
        text = runner.read_text(encoding="utf-8").lower()
        self.assertNotIn("q90_val2017", text)
        self.assertNotIn("common/coco/final", text)


if __name__ == "__main__":
    unittest.main()
