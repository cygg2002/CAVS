from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from board_client import validate_policy_binding


class PolicyBindingTest(unittest.TestCase):
    def test_accepts_matching_backend_and_rknn_model(self) -> None:
        policy = {"backend": "fp16", "rknn_model_sha256": "a" * 64}

        validate_policy_binding(policy, "fp16", "a" * 64)

    def test_rejects_policy_trained_for_another_backend(self) -> None:
        policy = {"backend": "int8-heads", "rknn_model_sha256": "a" * 64}

        with self.assertRaisesRegex(RuntimeError, "backend mismatch"):
            validate_policy_binding(policy, "fp16", "a" * 64)

    def test_rejects_policy_trained_for_another_rknn_model(self) -> None:
        policy = {"backend": "fp16", "rknn_model_sha256": "b" * 64}

        with self.assertRaisesRegex(RuntimeError, "RKNN model mismatch"):
            validate_policy_binding(policy, "fp16", "a" * 64)


if __name__ == "__main__":
    unittest.main()
