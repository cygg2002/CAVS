from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/cloudv2_core.py"


def load_core():
    spec = importlib.util.spec_from_file_location("cloudv2_core", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def protocol():
    import json

    return json.loads((ROOT / "configs/protocol.json").read_text())


class CloudV2CoreTest(unittest.TestCase):
    def test_protocol_freezes_only_cloud_model_and_seals_final_from_dev(self):
        core = load_core()
        checked = core.validate_protocol(protocol())
        self.assertEqual(checked["dev_roles"], ("fit", "calibration", "audit"))
        self.assertNotIn("final", checked["dev_roles"])
        self.assertEqual(checked["final_role"], "final")
        self.assertIs(checked["end2end"], False)

    def test_protocol_rejects_final_in_dev_roles(self):
        core = load_core()
        value = protocol()
        value["dev_roles"].append("final")
        with self.assertRaisesRegex(ValueError, "final"):
            core.validate_protocol(value)

    def test_protocol_rejects_noncanonical_inference_setting(self):
        core = load_core()
        value = protocol()
        value["inference"]["conf"] = 0.25
        with self.assertRaisesRegex(ValueError, "inference"):
            core.validate_protocol(value)

    def test_stage_roles_never_opens_final_during_dev(self):
        core = load_core()
        self.assertEqual(
            core.stage_roles(protocol(), "dev"), ("fit", "calibration", "audit")
        )
        self.assertEqual(core.stage_roles(protocol(), "final"), ("final",))
        with self.assertRaises(ValueError):
            core.stage_roles(protocol(), "unknown")

    def test_gpu_gate_uses_free_memory_not_utilization(self):
        core = load_core()
        self.assertTrue(
            core.gpu_has_capacity(
                total_mib=24564, used_mib=16647, minimum_free_mib=6144
            )
        )
        self.assertFalse(
            core.gpu_has_capacity(
                total_mib=24564, used_mib=20000, minimum_free_mib=6144
            )
        )

    def test_existing_link_must_resolve_to_frozen_source(self):
        import tempfile

        core = load_core()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text("{}")
            target = root / "target.json"
            core.ensure_symlink(source, target)
            core.ensure_symlink(source, target)
            wrong = root / "wrong.json"
            wrong.write_text("{}")
            target.unlink()
            target.symlink_to(wrong)
            with self.assertRaisesRegex(RuntimeError, "target drift"):
                core.ensure_symlink(source, target)


if __name__ == "__main__":
    unittest.main()
