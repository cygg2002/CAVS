from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from finalize_ena24_strict import (  # noqa: E402
    choose_final_oracle,
    completed_final_is_valid,
    file_sha256,
    pending_candidate_keys,
    validate_gate_lock,
)


class Ena24FinalizerTests(unittest.TestCase):
    def test_gate_lock_requires_matching_frozen_choice_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            choices = output / "strict_frozen_choices.json"
            choices.write_text(json.dumps([{"mode": "fixed_call"}]), encoding="utf-8")
            (output / "STRICT_RISK_GATE_COMPLETED").write_text(
                json.dumps(
                    {
                        "status": "strict_risk_gate_complete",
                        "strict_frozen_choices_sha256": file_sha256(choices),
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(validate_gate_lock(output), file_sha256(choices))

            choices.write_text(json.dumps([{"mode": "online"}]), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash"):
                validate_gate_lock(output)

    def test_final_oracle_uses_ap_then_kb_and_marks_diagnostic_only(self) -> None:
        rows = [
            {
                "mode": "fixed_call",
                "budget": 0.1,
                "ranker": "Confidence",
                "policy": "SharedAction",
                "official_map50_95": 0.7,
                "request_kb_per_image": 10.0,
            },
            {
                "mode": "fixed_call",
                "budget": 0.1,
                "ranker": "M2E",
                "policy": "GBDT-Hierarchical",
                "official_map50_95": 0.7,
                "request_kb_per_image": 9.0,
            },
        ]

        oracle = choose_final_oracle(
            rows,
            mode="fixed_call",
            budget=0.1,
            ranker_order=("Confidence", "M2E"),
            policy_order=("SharedAction", "GBDT-Hierarchical"),
        )

        self.assertEqual(oracle["ranker"], "M2E")
        self.assertTrue(oracle["diagnostic_only"])

    def test_pending_candidate_keys_are_deterministic_and_batch_limited(self) -> None:
        keys = [
            ("online", 0.25, "M2E", "SharedAction"),
            ("fixed_call", 0.1, "Confidence", "SharedAction"),
            ("fixed_byte", 0.05, "Confidence", "GBDT-Hierarchical"),
        ]

        pending = pending_candidate_keys(
            keys,
            completed={keys[1]},
            max_new_candidates=1,
        )

        self.assertEqual(pending, [keys[2]])

    def test_completed_final_requires_matching_integrity_and_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            frozen_sha = "a" * 64
            (output / "FINAL_COMPLETED").write_text("ok\n", encoding="ascii")
            (output / "final_integrity.json").write_text(
                json.dumps(
                    {
                        "strict_frozen_choices_sha256": frozen_sha,
                        "candidate_rows": 24,
                        "final_rows": 1,
                        "oracle_rows": 1,
                    }
                ),
                encoding="utf-8",
            )
            for name in (
                "final_candidate_results.csv",
                "final_results.csv",
                "final_oracle_diagnostic.csv",
                "final_selections.json",
            ):
                (output / name).write_text("present\n", encoding="utf-8")

            self.assertTrue(
                completed_final_is_valid(
                    output,
                    frozen_sha=frozen_sha,
                    expected_candidates=24,
                    expected_points=1,
                )
            )

            (output / "final_results.csv").unlink()
            self.assertFalse(
                completed_final_is_valid(
                    output,
                    frozen_sha=frozen_sha,
                    expected_candidates=24,
                    expected_points=1,
                )
            )


if __name__ == "__main__":
    unittest.main()
