from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from selector_core import validate_measured_e2e_claim  # noqa: E402


class SystemMeasurementContractTests(unittest.TestCase):
    def test_estimate_cannot_be_labeled_measured(self) -> None:
        with self.assertRaisesRegex(ValueError, "estimated"):
            validate_measured_e2e_claim(
                {
                    "latency_source": "estimated",
                    "edge_host": "mac",
                    "cloud_host": "server",
                }
            )

    def test_distinct_hosts_and_raw_logs_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "raw logs"):
            validate_measured_e2e_claim(
                {
                    "latency_source": "measured",
                    "edge_host": "mac",
                    "cloud_host": "server",
                    "edge_model_sha256": "a",
                    "cloud_model_sha256": "b",
                    "network_path": "physical_lan",
                    "sample_count": 100,
                }
            )

    def test_complete_real_measurement_metadata_passes(self) -> None:
        validate_measured_e2e_claim(
            {
                "latency_source": "measured",
                "edge_host": "mac",
                "cloud_host": "server",
                "edge_model_sha256": "a",
                "cloud_model_sha256": "b",
                "network_path": "physical_lan",
                "sample_count": 100,
                "raw_log_sha256": "c",
            }
        )


if __name__ == "__main__":
    unittest.main()
