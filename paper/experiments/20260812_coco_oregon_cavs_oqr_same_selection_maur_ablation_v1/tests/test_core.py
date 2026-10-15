import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("same_selection_maur_core", ROOT / "scripts/core.py")
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())


class SameSelectionMaurCoreTest(unittest.TestCase):
    def test_protocol_is_exact(self):
        CORE.validate_protocol(PROTOCOL)
        self.assertEqual(CORE.expected_point_rows(PROTOCOL), 48)
        self.assertEqual(CORE.expected_bootstrap_jobs(PROTOCOL), 9)

    def test_selection_key_matches_unified_contract(self):
        self.assertEqual(
            CORE.selection_key("final", "fixed_byte", 0.25),
            "final|CAVS-OQR|Joint-OQR|fixed_byte|0.25",
        )

    def test_transport_invariance_accepts_only_fusion_change(self):
        reversible = {
            "selected_images": 4,
            "original_count": 1,
            "q90_count": 2,
            "roi_count": 1,
            "request_bytes": 100,
            "kb_per_image": 0.1,
        }
        maur = {**reversible, "official_ap50_95": 0.5}
        CORE.validate_transport_pair(reversible, maur)
        with self.assertRaises(ValueError):
            CORE.validate_transport_pair(reversible, {**maur, "roi_count": 2})

    def test_fixed_call_zero_roi_requires_identical_metrics(self):
        left = {"roi_count": 0, "official_ap50_95": 0.5, "official_ap50": 0.7, "official_ap75": 0.55}
        CORE.validate_zero_roi_metrics(left, dict(left))
        with self.assertRaises(ValueError):
            CORE.validate_zero_roi_metrics(left, {**left, "official_ap75": 0.54})

    def test_group_predictions_is_deterministic(self):
        rows = [
            {"image_id": 2, "score": 0.2},
            {"image_id": 1, "score": 0.1},
            {"image_id": 2, "score": 0.3},
        ]
        self.assertEqual(list(CORE.group_predictions(rows)), [1, 2])
        self.assertEqual(len(CORE.group_predictions(rows)[2]), 2)


if __name__ == "__main__":
    unittest.main()
