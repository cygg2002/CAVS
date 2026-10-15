from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load("oric_moric_ac_core_test", ROOT / "scripts/core.py")
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


class CoreContractTest(unittest.TestCase):
    def test_protocol_is_paper_domain_and_has_36_training_jobs(self) -> None:
        CORE.validate_protocol(PROTOCOL)
        jobs = CORE.training_jobs(PROTOCOL)
        self.assertEqual(len(jobs), 36)
        self.assertEqual(len({job["id"] for job in jobs}), 36)
        self.assertTrue(all(job["action_space"] == PROTOCOL["action_space"] for job in jobs))

    def test_protocol_rejects_rknn_source_paths(self) -> None:
        changed = json.loads(json.dumps(PROTOCOL))
        changed["source_factorial_experiment"] = "paper/experiments/rknn_fp16"
        with self.assertRaisesRegex(ValueError, "forbidden deployment-domain"):
            CORE.validate_protocol(changed)

    def test_unified_cdf_uses_all_valid_cloud_actions_and_preserves_ties(self) -> None:
        rewards = np.asarray(
            [[0.0, 1.0, 3.0, 3.0], [0.0, 2.0, 4.0, 99.0]], dtype=np.float64
        )
        action_type = np.asarray([[0, 1, 1, 2], [0, 1, 1, 2]], dtype=np.int64)
        action_mask = np.asarray([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=bool)
        transformed, reference = CORE.unified_cloud_cdf_targets(
            rewards, action_type, action_mask, reference_rows=[0, 1]
        )
        np.testing.assert_allclose(reference, [1.0, 2.0, 3.0, 3.0, 4.0])
        self.assertEqual(transformed[0, 0], 0.0)
        self.assertEqual(transformed[1, 3], 0.0)
        self.assertEqual(transformed[0, 2], transformed[0, 3])
        self.assertAlmostEqual(transformed[0, 2], 0.8)

    def test_cdf_reference_can_exclude_validation_rows(self) -> None:
        rewards = np.asarray([[0.0, 1.0], [0.0, 100.0]], dtype=np.float64)
        action_type = np.asarray([[0, 1], [0, 1]], dtype=np.int64)
        action_mask = np.ones_like(action_type, dtype=bool)
        transformed, reference = CORE.unified_cloud_cdf_targets(
            rewards, action_type, action_mask, reference_rows=[0]
        )
        np.testing.assert_allclose(reference, [1.0])
        self.assertEqual(transformed[0, 1], 1.0)
        self.assertEqual(transformed[1, 1], 1.0)


if __name__ == "__main__":
    unittest.main()
