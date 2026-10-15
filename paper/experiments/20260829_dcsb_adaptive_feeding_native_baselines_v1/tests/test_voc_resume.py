from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VOC = load_module("test_dcsb_af_voc_resume", ROOT / "scripts/run_voc_resume.py")


class VocHashBindingTests(unittest.TestCase):
    def test_reads_detector_hash_from_source_protocol_not_role_manifest(self):
        protocol = {
            "detectors": {"cloud": {"weights_sha256": "frozen-cloud-hash"}}
        }
        self.assertEqual(VOC.voc_cloud_hash(protocol), "frozen-cloud-hash")


if __name__ == "__main__":
    unittest.main()

