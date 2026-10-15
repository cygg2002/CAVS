from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


HOST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST))

from build_run_matrix import MATRIX_SCHEMA, build_run_matrix


def write(path: Path, value: str = "asset") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


class BuildRunMatrixTest(unittest.TestCase):
    def _bundle(self, root: Path) -> tuple[Path, Path]:
        bundle = root / "board_bundle"
        protocol = {
            "protocol_id": "coco_rknn_fp16_causal_online_hil_final_v2",
            "dataset": {"expected_images": 5000},
            "budgets": [0.1, 0.25, 0.4],
            "methods": {
                "edge_only": {"policy": "scripts/online_methods.py"},
                "always_original": {"policy": "scripts/online_methods.py"},
                "cavs_oq": {"policy": "policies/cavs_oq/manifest.json"},
                "cavs_oqr": {"policy": "policies/cavs_oqr/manifest.json"},
                "oric": {"policy": "policies/external/manifest.json"},
                "moric": {"policy": "policies/external/manifest.json"},
                "confidence": {"policy": "scripts/online_methods.py"},
                "random": {"policy": "scripts/online_methods.py"},
            },
        }
        protocol_path = write(
            bundle / "configs/protocol.json",
            json.dumps(protocol, sort_keys=True),
        )
        write(bundle / "scripts/online_methods.py")
        write(bundle / "policies/cavs_oq/manifest.json")
        write(bundle / "policies/cavs_oqr/manifest.json")
        write(bundle / "policies/external/manifest.json")
        for method in ("cavs_oq", "cavs_oqr", "oric", "moric", "confidence"):
            for budget in (10, 25, 40):
                write(bundle / f"controls/{method}/{budget}.json")
        for method in ("cavs_oq", "cavs_oqr"):
            for budget in (10, 25, 40):
                write(bundle / f"controls/{method}_static/{budget}.json")
        for seed in (20260720, 20260721, 20260722, 20260723, 20260724):
            for budget in (10, 25, 40):
                write(bundle / f"controls/random_seed_{seed}/{budget}.json")
        write(bundle / "controls/edge_only/100.json")
        write(bundle / "controls/always_original/100.json")
        return bundle, protocol_path

    def test_freezes_complete_unique_paper_matrix_and_hashes(self) -> None:
        with TemporaryDirectory() as directory:
            bundle, protocol = self._bundle(Path(directory))
            matrix = build_run_matrix(bundle, protocol)

        self.assertEqual(matrix["schema"], MATRIX_SCHEMA)
        self.assertEqual(matrix["protocol_id"], "coco_rknn_fp16_causal_online_hil_final_v2")
        self.assertEqual(matrix["expected_images"], 5000)
        self.assertEqual(matrix["ordering_seed"], 20260828)
        self.assertEqual(matrix["random_seeds"], [20260720, 20260721, 20260722, 20260723, 20260724])
        self.assertEqual(matrix["family_counts"], {
            "controller_ablation": 6,
            "main": 32,
            "network_sensitivity": 8,
        })
        self.assertEqual(len(matrix["runs"]), 46)
        keys = [row["run_key"] for row in matrix["runs"]]
        self.assertEqual(len(keys), len(set(keys)))
        for row in matrix["runs"]:
            self.assertRegex(row["policy_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(row["control_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(row["expected_images"], 5000)
            self.assertFalse(row["global_plan"])
            self.assertTrue(row["zero_lookahead"])
            self.assertEqual(row["buffer_frames"], 0)

        main = [row for row in matrix["runs"] if row["family"] == "main"]
        self.assertEqual(sum(row["method"] == "random" for row in main), 15)
        self.assertEqual(sum(row["method"] == "edge_only" for row in main), 1)
        self.assertEqual(sum(row["method"] == "always_original" for row in main), 1)
        network = [row for row in matrix["runs"] if row["family"] == "network_sensitivity"]
        self.assertEqual({row["network_profile"] for row in network}, {
            "wan_20ms_20mbps", "wan_50ms_5mbps",
        })

    def test_is_byte_deterministic_and_rejects_missing_assets(self) -> None:
        with TemporaryDirectory() as directory:
            bundle, protocol = self._bundle(Path(directory))
            first = build_run_matrix(bundle, protocol)
            second = build_run_matrix(bundle, protocol)
            self.assertEqual(
                hashlib.sha256(json.dumps(first, sort_keys=True).encode()).hexdigest(),
                hashlib.sha256(json.dumps(second, sort_keys=True).encode()).hexdigest(),
            )
            (bundle / "policies/cavs_oq/manifest.json").unlink()
            with self.assertRaises(FileNotFoundError):
                build_run_matrix(bundle, protocol)


if __name__ == "__main__":
    unittest.main()
