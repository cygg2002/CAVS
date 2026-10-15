from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_system_measurement import (  # noqa: E402
    build_feasibility_report,
    render_status_note,
    write_artifacts,
)


class SystemMeasurementAuditTests(unittest.TestCase):
    def test_report_records_feasibility_without_measured_e2e_claim(self) -> None:
        report = build_feasibility_report(REPO_ROOT)

        self.assertTrue(report["topology"]["distinct_hosts_feasible"])
        self.assertEqual(report["topology"]["edge"]["processor"], "Apple M2 Pro")
        self.assertEqual(
            report["topology"]["cloud"]["accelerator"],
            "NVIDIA GeForce RTX 4090",
        )
        self.assertEqual(report["topology"]["cloud"]["availability"], "shared_busy")
        self.assertEqual(
            report["network"]["controlled_shaping"]["status"],
            "privilege_required",
        )
        self.assertTrue(report["implementation"]["edge_client"]["exists"])
        self.assertTrue(report["implementation"]["cloud_server"]["exists"])
        self.assertFalse(report["claim_gate"]["measured_e2e_claim_allowed"])
        self.assertEqual(report["claim_gate"]["status"], "blocked")
        self.assertIn(
            "quiet_or_exclusive_cloud_gpu",
            report["claim_gate"]["blocking_requirements"],
        )
        self.assertIn(
            "frozen_payload_replay_manifest",
            report["claim_gate"]["blocking_requirements"],
        )

    def test_report_contains_no_credentials_or_measured_values(self) -> None:
        report = build_feasibility_report(REPO_ROOT)
        serialized = json.dumps(report, ensure_ascii=False).lower()

        self.assertNotIn("password", serialized)
        self.assertNotIn("wcy111", serialized)
        self.assertNotIn("latency_ms", serialized)
        self.assertNotIn("throughput_fps", serialized)
        self.assertNotIn("energy_j", serialized)

    def test_status_note_states_the_blocker_and_next_contract(self) -> None:
        note = render_status_note(build_feasibility_report(REPO_ROOT))

        self.assertIn("不能声称已实测 E2E", note)
        self.assertIn("Apple M2 Pro", note)
        self.assertIn("NVIDIA GeForce RTX 4090", note)
        self.assertIn("空闲或独占", note)
        self.assertIn("冻结权重", note)
        self.assertIn("payload 回放", note)
        self.assertIn("原始日志", note)

    def test_write_artifacts_uses_only_requested_output_directory(self) -> None:
        report = build_feasibility_report(REPO_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            json_path, note_path = write_artifacts(report, output_root)

            self.assertEqual(json_path, output_root / "runs/system/feasibility.json")
            self.assertEqual(
                note_path,
                output_root / "notes/system_measurement_status_zh.md",
            )
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8"))["claim_gate"][
                    "status"
                ],
                "blocked",
            )
            self.assertIn(
                "不能声称已实测 E2E",
                note_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
