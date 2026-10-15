from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


EXPERIMENT = Path(__file__).resolve().parents[2]
FORMAL = EXPERIMENT / "board_formal"
sys.path.insert(0, str(FORMAL))

from remaining_matrix import network_commands, validate_remaining_matrix


class RemainingMatrixTest(unittest.TestCase):
    def test_validates_frozen_remaining_rows(self) -> None:
        rows = validate_remaining_matrix(
            EXPERIMENT / "RUN_MATRIX.json",
            EXPERIMENT / "board_bundle",
            EXPERIMENT / "board_bundle/configs/protocol.json",
        )
        self.assertEqual([row["run_index"] for row in rows], list(range(33, 47)))
        self.assertEqual(
            {family: sum(row["family"] == family for row in rows) for family in {
                "controller_ablation", "network_sensitivity"
            }},
            {"controller_ablation": 6, "network_sensitivity": 8},
        )

    def test_rejects_changed_remaining_family(self) -> None:
        source = json.loads((EXPERIMENT / "RUN_MATRIX.json").read_text(encoding="utf-8"))
        source["runs"][32]["family"] = "main"
        with TemporaryDirectory() as directory:
            matrix = Path(directory) / "RUN_MATRIX.json"
            matrix.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "indices|counts"):
                validate_remaining_matrix(
                    matrix,
                    EXPERIMENT / "board_bundle",
                    EXPERIMENT / "board_bundle/configs/protocol.json",
                )

    def test_builds_native_and_shaped_tc_commands(self) -> None:
        self.assertEqual(network_commands("wlan0", "native", {}), [])
        commands = network_commands(
            "wlan0",
            "wan_20ms_20mbps",
            {
                "mechanism": "linux_tc_netem_tbf",
                "added_round_trip_delay_ms": 20,
                "uplink_mbps": 20,
                "packet_loss_percent": 0.0,
            },
        )
        self.assertEqual(
            commands,
            [
                [
                    "tc", "qdisc", "replace", "dev", "wlan0", "root", "handle", "1:",
                    "tbf", "rate", "20mbit", "burst", "16kb", "latency", "2000ms",
                ],
                [
                    "tc", "qdisc", "replace", "dev", "wlan0", "parent", "1:1",
                    "handle", "10:", "netem", "delay", "20ms", "loss", "0%",
                    "limit", "10000",
                ],
            ],
        )

    def test_rejects_unregistered_shaping_mechanism(self) -> None:
        with self.assertRaisesRegex(ValueError, "mechanism"):
            network_commands(
                "wlan0",
                "bad",
                {
                    "mechanism": "userspace_sleep",
                    "added_round_trip_delay_ms": 20,
                    "uplink_mbps": 20,
                    "packet_loss_percent": 0.0,
                },
            )


if __name__ == "__main__":
    unittest.main()
