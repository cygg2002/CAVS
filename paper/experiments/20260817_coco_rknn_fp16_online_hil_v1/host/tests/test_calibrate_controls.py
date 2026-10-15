from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
HOST = ROOT / "host"
BOARD_SCRIPTS = ROOT / "board_bundle" / "scripts"
sys.path.insert(0, str(HOST))
sys.path.insert(0, str(BOARD_SCRIPTS))

from calibrate_controls import calibrate_control, load_calibration_inputs
from control_schema import CONTROL_SCHEMA, validate_formal_control


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CalibrateControlsTest(unittest.TestCase):
    def _bundle(self, *, split: str = "calibration") -> dict:
        return {
            "schema": "cavs_online_calibration_records_v1",
            "protocol_id": "test_online_v1",
            "source_split": split,
            "method": "cavs_oqr",
            "calibration_objective": {
                "name": "teacher_contextual_ap50_95",
                "direction": "maximize",
            },
            "action_ids": ["edge", "original", "q90"],
            "source_hashes": {"policy": "a" * 64, "sequence": "b" * 64},
            "records": [
                {
                    "image_id": 1,
                    "payload_bytes": {"edge": 0, "original": 100, "q90": 40},
                    "scores": {"edge": 0.0, "original": 1.0, "q90": 0.75},
                    "objective_values": {"edge": 0.0, "original": 1.0, "q90": 1.0},
                }
            ],
        }

    def test_grid_order_and_duplicates_do_not_change_deterministic_tie_break(self) -> None:
        bundle = self._bundle()

        first = calibrate_control(
            bundle,
            budget="1",
            lambda_grid=[1.0, 0.5, -0.0, 0.5],
            calibration_records_sha256="c" * 64,
            protocol_sha256="d" * 64,
        )
        second = calibrate_control(
            bundle,
            budget="1.0",
            lambda_grid=[0.0, 0.5, 1.0],
            calibration_records_sha256="c" * 64,
            protocol_sha256="d" * 64,
        )

        self.assertEqual(first, second)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(first["lambda"], 0.5)
        self.assertEqual(first["median_original_bytes"], 100.0)
        self.assertEqual(first["diagnostics"]["action_counts"], {"q90": 1})
        self.assertEqual(first["calibration_objective"]["value"], 1.0)
        self.assertEqual(first["lambda_grid"], [0.0, 0.5, 1.0])

    def test_causal_replay_never_spends_future_credit(self) -> None:
        bundle = self._bundle()
        bundle["records"] = [
            {
                "image_id": 1,
                "payload_bytes": {"edge": 0, "original": 100, "q90": 60},
                "scores": {"edge": 0.0, "original": 2.0, "q90": 1.0},
                "objective_values": {"edge": 0.0, "original": 10.0, "q90": 5.0},
            },
            {
                "image_id": 2,
                "payload_bytes": {"edge": 0, "original": 100, "q90": 60},
                "scores": {"edge": 0.0, "original": 2.0, "q90": 1.0},
                "objective_values": {"edge": 0.0, "original": 10.0, "q90": 5.0},
            },
        ]

        control = calibrate_control(
            bundle,
            budget="1/2",
            lambda_grid=[0.0],
            calibration_records_sha256="c" * 64,
            protocol_sha256="d" * 64,
        )

        self.assertEqual(control["diagnostics"]["action_counts"], {"edge": 1, "original": 1})
        self.assertEqual(control["diagnostics"]["spent_bytes"], 100)
        self.assertEqual(control["diagnostics"]["maximum_prefix_violation_bytes"], 0)

    def test_final_or_undeclared_objective_is_rejected(self) -> None:
        final_bundle = self._bundle(split="final")
        with self.assertRaisesRegex(ValueError, "calibration"):
            calibrate_control(
                final_bundle,
                budget="0.1",
                lambda_grid=[0.0],
                calibration_records_sha256="c" * 64,
                protocol_sha256="d" * 64,
            )

        bundle = self._bundle()
        bundle["calibration_objective"] = {"name": "teacher_contextual_ap50_95"}
        with self.assertRaisesRegex(ValueError, "direction"):
            calibrate_control(
                bundle,
                budget="0.1",
                lambda_grid=[0.0],
                calibration_records_sha256="c" * 64,
                protocol_sha256="d" * 64,
            )

    def test_source_files_and_protocol_hashes_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.bin"
            sequence = root / "sequence.json"
            protocol = root / "protocol.json"
            records = root / "records.json"
            policy.write_bytes(b"policy-v1")
            sequence.write_bytes(b"sequence-v1")
            protocol.write_text(json.dumps({"protocol_id": "test_online_v1"}), encoding="utf-8")
            bundle = self._bundle()
            bundle["source_hashes"] = {
                "policy": _sha256(policy),
                "sequence": _sha256(sequence),
            }
            records.write_text(json.dumps(bundle), encoding="utf-8")

            loaded, metadata = load_calibration_inputs(
                records_path=records,
                protocol_path=protocol,
                source_files={"policy": policy, "sequence": sequence},
            )
            self.assertEqual(loaded, bundle)
            self.assertEqual(metadata["calibration_records"], _sha256(records))
            self.assertEqual(metadata["protocol"], _sha256(protocol))

            policy.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "policy.*SHA-256"):
                load_calibration_inputs(
                    records_path=records,
                    protocol_path=protocol,
                    source_files={"policy": policy, "sequence": sequence},
                )

    def test_cli_writes_a_formal_control_that_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.bin"
            sequence = root / "sequence.json"
            protocol = root / "protocol.json"
            records = root / "records.json"
            output_a = root / "control-a.json"
            output_b = root / "control-b.json"
            policy.write_bytes(b"policy-v1")
            sequence.write_bytes(b"sequence-v1")
            protocol.write_text(json.dumps({"protocol_id": "test_online_v1"}), encoding="utf-8")
            bundle = self._bundle()
            bundle["source_hashes"] = {
                "policy": _sha256(policy),
                "sequence": _sha256(sequence),
            }
            records.write_text(json.dumps(bundle), encoding="utf-8")
            command = [
                sys.executable,
                str(HOST / "calibrate_controls.py"),
                "--records", str(records),
                "--protocol", str(protocol),
                "--method", "cavs_oqr",
                "--budget", "1",
                "--objective", "teacher_contextual_ap50_95",
                "--lambda-grid", "1,0.5,0,0.5",
                "--source", f"policy={policy}",
                "--source", f"sequence={sequence}",
            ]
            subprocess.run(command + ["--output", str(output_a)], check=True)
            subprocess.run(command + ["--output", str(output_b)], check=True)

            self.assertEqual(output_a.read_bytes(), output_b.read_bytes())
            control = json.loads(output_a.read_text(encoding="utf-8"))
            self.assertEqual(control["schema"], CONTROL_SCHEMA)
            self.assertEqual(control["protocol_id"], "test_online_v1")
            self.assertEqual(control["method"], "cavs_oqr")
            self.assertEqual(control["source_split"], "calibration")
            self.assertEqual(control["source_hashes"]["policy"], _sha256(policy))
            validate_formal_control(
                control,
                expected_protocol_id="test_online_v1",
                expected_method="cavs_oqr",
                expected_budget="1",
            )

    def test_formal_control_validator_rejects_non_calibration_source(self) -> None:
        control = calibrate_control(
            self._bundle(),
            budget="1",
            lambda_grid=[0.5],
            calibration_records_sha256="c" * 64,
            protocol_sha256="d" * 64,
        )
        control["source_split"] = "final"

        with self.assertRaisesRegex(ValueError, "calibration"):
            validate_formal_control(control)

    def test_fractional_payload_bytes_are_not_silently_truncated(self) -> None:
        bundle = self._bundle()
        bundle["records"][0]["payload_bytes"]["q90"] = 40.5

        with self.assertRaisesRegex(ValueError, "integer"):
            calibrate_control(
                bundle,
                budget="1",
                lambda_grid=[0.0],
                calibration_records_sha256="c" * 64,
                protocol_sha256="d" * 64,
            )

    def test_formal_control_rejects_a_final_tuning_claim(self) -> None:
        control = calibrate_control(
            self._bundle(),
            budget="1",
            lambda_grid=[0.5],
            calibration_records_sha256="c" * 64,
            protocol_sha256="d" * 64,
        )
        control["final_used_for_tuning"] = True

        with self.assertRaisesRegex(ValueError, "final"):
            validate_formal_control(control)

    def test_random_control_binds_the_calibration_seed(self) -> None:
        bundle = self._bundle()
        bundle["method"] = "random"
        bundle["random_seed"] = 20260720
        bundle["action_ids"] = ["edge", "original"]
        for record in bundle["records"]:
            record["payload_bytes"].pop("q90")
            record["scores"].pop("q90")
            record["objective_values"].pop("q90")

        control = calibrate_control(
            bundle,
            budget="1",
            lambda_grid=[0.5],
            calibration_records_sha256="c" * 64,
            protocol_sha256="d" * 64,
        )

        self.assertEqual(control["random_seed"], 20260720)
        validate_formal_control(control, expected_method="random")


if __name__ == "__main__":
    unittest.main()
