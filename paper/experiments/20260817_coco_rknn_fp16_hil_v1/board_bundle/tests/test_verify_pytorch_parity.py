from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_pytorch_parity import configure_reference_identity


class ReferenceIdentityTest(unittest.TestCase):
    def test_configures_reference_for_the_bound_fp16_rknn_model(self) -> None:
        source = SimpleNamespace(
            EXPECTED_EDGE_MODEL="yolo26n.pt", EXPECTED_EDGE_MODEL_SHA256="b" * 64
        )
        manifest = {
            "backend": "fp16",
            "rknn_model_sha256": "a" * 64,
            "edge_model": {"name": "edge.rknn", "sha256": "a" * 64},
        }

        configure_reference_identity(source, manifest)

        self.assertEqual(source.EXPECTED_EDGE_MODEL, "edge.rknn")
        self.assertEqual(source.EXPECTED_EDGE_MODEL_SHA256, "a" * 64)

    def test_rejects_unbound_reference_identity(self) -> None:
        source = SimpleNamespace()
        manifest = {
            "backend": "fp16",
            "rknn_model_sha256": "a" * 64,
            "edge_model": {"name": "edge.rknn", "sha256": "b" * 64},
        }

        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            configure_reference_identity(source, manifest)


if __name__ == "__main__":
    unittest.main()
