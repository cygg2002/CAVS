import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "voc_torchvision_gate_runner", ROOT / "scripts/run_gates_when_gpu_free.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ResourceGateOverrideTest(unittest.TestCase):
    def test_start_now_skips_resource_wait(self):
        original = MODULE.wait_for_gpu
        waiter = Mock()
        MODULE.wait_for_gpu = waiter
        try:
            MODULE.acquire_gpu(start_now=True)
            waiter.assert_not_called()
            MODULE.acquire_gpu(start_now=False)
            waiter.assert_called_once_with()
        finally:
            MODULE.wait_for_gpu = original


if __name__ == "__main__":
    unittest.main()
