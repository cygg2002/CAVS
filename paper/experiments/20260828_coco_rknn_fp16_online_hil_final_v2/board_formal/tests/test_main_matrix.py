from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


FORMAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORMAL))

from main_matrix import gate_run, validate_main_matrix


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MainMatrixTest(unittest.TestCase):
    def _matrix_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        bundle = root / "board_bundle"
        policy = bundle / "policy.json"
        control = bundle / "control.json"
        protocol = bundle / "configs/protocol.json"
        for path, value in (
            (policy, {}),
            (control, {}),
            (
                protocol,
                {
                    "protocol_id": "coco_rknn_fp16_causal_online_hil_final_v2",
                    "dataset": {"expected_images": 5000},
                },
            ),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")

        specs: list[tuple[str, int, int | None]] = [
            ("edge_only", 100, None),
            ("always_original", 100, None),
        ]
        for method in ("cavs_oq", "cavs_oqr", "oric", "moric", "confidence"):
            specs.extend((method, budget, None) for budget in (10, 25, 40))
        for seed in (20260720, 20260721, 20260722, 20260723, 20260724):
            specs.extend(("random", budget, seed) for budget in (10, 25, 40))
        rows = []
        for index, (method, percent, random_seed) in enumerate(specs, start=1):
            fraction = Fraction(1, 1) if percent == 100 else {
                10: Fraction(1, 10),
                25: Fraction(1, 4),
                40: Fraction(2, 5),
            }[percent]
            rows.append(
                {
                    "run_index": index,
                    "run_key": f"main-{method}-{percent}-{random_seed}",
                    "family": "main",
                    "method": method,
                    "budget_percent": percent,
                    "budget_numerator": fraction.numerator,
                    "budget_denominator": fraction.denominator,
                    "random_seed": random_seed,
                    "controller_variant": "registered",
                    "network_profile": "native",
                    "policy": "policy.json",
                    "policy_sha256": sha256(policy),
                    "control": "control.json",
                    "control_sha256": sha256(control),
                    "expected_images": 5000,
                    "zero_lookahead": True,
                    "global_plan": False,
                    "buffer_frames": 0,
                }
            )
        matrix = {
            "schema": "cavs_online_paper_run_matrix_v2",
            "protocol_id": "coco_rknn_fp16_causal_online_hil_final_v2",
            "protocol_sha256": sha256(protocol),
            "expected_images": 5000,
            "family_counts": {"main": 32},
            "random_seeds": [20260720, 20260721, 20260722, 20260723, 20260724],
            "final_used_for_matrix_selection": False,
            "runs": rows,
        }
        matrix_path = root / "RUN_MATRIX.json"
        matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
        return bundle, protocol, matrix_path

    def test_validates_complete_hash_pinned_main_matrix(self) -> None:
        with TemporaryDirectory() as directory:
            bundle, protocol, matrix = self._matrix_fixture(Path(directory))
            rows = validate_main_matrix(matrix, bundle, protocol)
        self.assertEqual(len(rows), 32)
        self.assertEqual([row["run_index"] for row in rows], list(range(1, 33)))

    def test_rejects_asset_tampering(self) -> None:
        with TemporaryDirectory() as directory:
            bundle, protocol, matrix = self._matrix_fixture(Path(directory))
            (bundle / "policy.json").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "policy SHA-256"):
                validate_main_matrix(matrix, bundle, protocol)

    def test_gates_complete_causal_run_and_rejects_prefix_violation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            contract = {
                "protocol_id": "coco_rknn_fp16_causal_online_hil_final_v2",
                "method": "cavs_oq",
                "budget_numerator": 1,
                "budget_denominator": 4,
                "image_count": 3,
                "random_seed": None,
                "zero_lookahead": True,
                "global_plan": False,
                "buffer_frames": 0,
                "policy_sha256": "a" * 64,
                "control_sha256": "b" * 64,
            }
            (root / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
            rows = []
            spent = 0
            cumulative = 0
            for position, payload in enumerate((0, 20, 0)):
                cumulative += 100
                spent += payload
                rows.append(
                    {
                        "position": position,
                        "status": "committed",
                        "execution_status": "not_required" if payload == 0 else "ok",
                        "decision": {
                            "action": "edge" if payload == 0 else "q90",
                            "payload_bytes": payload,
                            "original_bytes": 100,
                            "cumulative_original_bytes": cumulative,
                            "spent_bytes": spent,
                        },
                        "controller_state": {
                            "budget_numerator": 1,
                            "budget_denominator": 4,
                        },
                        "cloud": {"request_body_bytes": payload},
                        "shape": [100, 100],
                        "resolved_boxes": [
                            {"xyxy": [1.0, 2.0, 5.0, 7.0], "conf": 0.5, "cls": 1}
                        ],
                    }
                )
            frames = root / "frames.jsonl"
            frames.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            matrix_row = {
                "method": "cavs_oq",
                "budget_numerator": 1,
                "budget_denominator": 4,
                "random_seed": None,
                "policy_sha256": "a" * 64,
                "control_sha256": "b" * 64,
                "expected_images": 3,
            }
            summary = gate_run(frames, root / "run_contract.json", matrix_row)
            self.assertEqual(summary["frames"], 3)
            self.assertEqual(summary["maximum_prefix_violation_bytes"], 0)
            rows[0]["decision"].update(action="q90", payload_bytes=30, spent_bytes=30)
            rows[0]["execution_status"] = "ok"
            rows[0]["cloud"]["request_body_bytes"] = 30
            rows[1]["decision"]["spent_bytes"] = 50
            rows[2]["decision"]["spent_bytes"] = 50
            frames.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prefix"):
                gate_run(frames, root / "run_contract.json", matrix_row)


if __name__ == "__main__":
    unittest.main()
