from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "repair_bootstrap_metadata", ROOT / "scripts/repair_bootstrap_metadata.py"
)
assert SPEC is not None and SPEC.loader is not None
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


class RepairBootstrapMetadataTest(unittest.TestCase):
    def test_repairs_labels_without_changing_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "ena24"
            track_b = dataset_root / "track_b"
            bootstrap = track_b / "bootstrap"
            bootstrap.mkdir(parents=True)
            core_fields = [
                "action_policy",
                "nominal_budget",
                "method",
                "mode",
                "comparator",
            ]
            with (track_b / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=core_fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "action_policy": "SharedAction",
                        "nominal_budget": "0.1",
                        "method": "DualHead-Value",
                        "mode": "fixed_call",
                        "comparator": "HistoricalRanker",
                    }
                )
            payload = {
                "dataset": "ena24",
                "key": "SharedAction|0.1",
                "comparison": "Confidence-Residual M2E - Confidence",
                "mode": "fixed_call",
                "nominal_budget": 0.25,
                "replicates": 200,
                "official_map50_95_delta_mean": 0.0123,
                "official_map50_95_delta_ci_low": -0.001,
                "official_map50_95_delta_ci_high": 0.02,
                "official_map50_95_probability_gt_zero": 0.8,
                "request_kb_per_image_delta_mean": -1.5,
                "request_kb_per_image_delta_ci_low": -2.0,
                "request_kb_per_image_delta_ci_high": -1.0,
                "action_pair_sha256": "pair-hash",
            }
            artifact = bootstrap / "SharedAction_0.1.json"
            artifact.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            marker = bootstrap / "SharedAction_0.1.completion.json"
            marker.write_text(
                json.dumps({"complete": True, "sha256": "old-hash"}) + "\n",
                encoding="utf-8",
            )
            aggregate_fields = list(payload)
            with (track_b / "paired_bootstrap.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
                writer.writeheader()
                writer.writerow(payload)

            report = repair.repair_dataset(dataset_root, dry_run=False)
            self.assertTrue(report["numeric_fields_unchanged"])
            self.assertTrue(report["numeric_json_fields_unchanged"])
            fixed = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(fixed["comparison"], "DualHead-Value - HistoricalRanker")
            self.assertEqual(fixed["nominal_budget"], 0.1)
            self.assertEqual(fixed["action_pair_sha256"], "pair-hash")
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(
                marker_payload["sha256"],
                hashlib.sha256(artifact.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
