import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "voc_endpoint_gate_under_test", ROOT / "scripts/evaluate_endpoints.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EndpointGateTest(unittest.TestCase):
    def test_published_ap_gate_accepts_small_and_rejects_large_drift(self):
        self.assertAlmostEqual(MODULE.published_ap_drift(0.668, 0.667), -0.001)
        MODULE.require_published_ap_consistency(0.668, 0.667, tolerance=0.02)
        with self.assertRaises(RuntimeError):
            MODULE.require_published_ap_consistency(0.668, 0.620, tolerance=0.02)


if __name__ == "__main__":
    unittest.main()
