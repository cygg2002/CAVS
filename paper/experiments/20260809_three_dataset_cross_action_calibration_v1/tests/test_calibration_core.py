from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibration_core import (  # noqa: E402
    METHODS,
    apply_isotonic,
    assert_fixed_membership,
    build_fixed_membership_groups,
    crossfit_isotonic,
    evaluate_gate,
    fit_isotonic,
    restrict_to_oq_values,
    validate_action_availability,
    validate_protocol,
    validate_result_rows,
)


class CalibrationCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_protocol_is_frozen(self) -> None:
        validate_protocol(self.protocol)
        self.assertEqual(len(METHODS), 5)

    def test_protocol_rejects_final_access(self) -> None:
        changed = json.loads(json.dumps(self.protocol))
        changed["final_accessed"] = True
        with self.assertRaises(ValueError):
            validate_protocol(changed)

    def test_isotonic_prediction_is_monotone(self) -> None:
        x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.asarray([0.0, 3.0, 1.0, 4.0, 2.0, 5.0])
        model = fit_isotonic(x, y, bins=6)
        predicted = apply_isotonic(model, np.linspace(-1.0, 6.0, 101))
        self.assertTrue(np.all(np.diff(predicted) >= -1e-12))

    def test_crossfit_never_trains_on_heldout_indices(self) -> None:
        scores = np.arange(12, dtype=np.float64)
        targets = scores * 0.5
        folds = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
        predicted, reports = crossfit_isotonic(scores, targets, folds, bins=4)
        self.assertTrue(np.isfinite(predicted).all())
        for report in reports:
            self.assertFalse(set(report["train_indices"]) & set(report["heldout_indices"]))
        self.assertEqual(sorted(i for r in reports for i in r["heldout_indices"]), list(range(12)))

    def test_crossfit_ignores_missing_action_targets_without_imputing_zero(self) -> None:
        scores = np.arange(8, dtype=np.float64)
        targets = np.asarray([0.0, np.nan, 1.0, np.nan, 2.0, 2.5, 3.0, 4.0])
        valid = np.isfinite(targets)
        folds = [[0, 1, 4, 5], [2, 3, 6, 7]]
        predicted, reports = crossfit_isotonic(
            scores, targets, folds, bins=3, valid_mask=valid
        )
        self.assertTrue(np.isfinite(predicted).all())
        for report in reports:
            self.assertTrue(set(report["train_indices"]) <= set(np.flatnonzero(valid)))

    def test_fixed_membership_must_be_exact(self) -> None:
        assert_fixed_membership({1: "q90", 2: "original"}, {1, 2})
        with self.assertRaises(RuntimeError):
            assert_fixed_membership({1: "q90"}, {1, 2})

    def test_fixed_membership_groups_have_no_edge_escape(self) -> None:
        samples = [
            {
                "image_id": 1,
                "available": {"original": True, "q90": True},
                "payload_bytes": {"original": 100, "q90": 30},
            },
            {
                "image_id": 2,
                "available": {"original": True, "q90": True},
                "payload_bytes": {"original": 120, "q90": 40},
            },
        ]
        values = {
            1: {"original": 0.5, "q90": 0.4},
            2: {"original": 0.2, "q90": 0.3},
        }
        groups = build_fixed_membership_groups(samples, {1, 2}, values)
        self.assertEqual([[a["action_id"] for a in group] for group in groups], [["original", "q90"], ["original", "q90"]])
        self.assertNotIn("edge", {a["action_id"] for group in groups for a in group})

    def test_oq_teacher_values_exclude_roi_and_edge(self) -> None:
        values = {
            1: {"edge": 0.0, "original": 0.2, "q90": 0.1, "roi": 9.0},
            2: {"original": 0.3},
        }
        restricted = restrict_to_oq_values(values)
        self.assertEqual(
            restricted,
            {1: {"original": 0.2, "q90": 0.1}, 2: {"original": 0.3}},
        )
        self.assertFalse(any("roi" in actions for actions in restricted.values()))

    def test_preflight_accepts_original_only_when_q90_is_unavailable(self) -> None:
        samples = [
            {
                "image_id": 61925,
                "available": {"original": True, "q90": False},
                "payload_bytes": {"original": 100, "q90": None},
            }
        ]
        validate_action_availability(samples)

    def test_result_matrix_and_capacity_are_enforced(self) -> None:
        rows = []
        for role in self.protocol["roles"]:
            for method in self.protocol["methods"]:
                for mode in self.protocol["modes"]:
                    for budget in self.protocol["budgets"]:
                        rows.append(
                            {
                                "dataset": "ena24",
                                "role": role,
                                "method": method,
                                "mode": mode,
                                "budget": budget,
                                "capacity_ok": True,
                                "selected_images": int(math.floor((1400 if role == "fit_oof" else 400) * budget + 0.5)),
                            }
                        )
        validate_result_rows(rows, self.protocol, "ena24")
        rows[0]["capacity_ok"] = False
        with self.assertRaises(ValueError):
            validate_result_rows(rows, self.protocol, "ena24")

    def test_gate_requires_cross_dataset_support(self) -> None:
        rows = []
        for dataset in self.protocol["datasets"]:
            for budget in self.protocol["budgets"]:
                raw = 0.70 if dataset == "ena24" else 0.50
                calibrated = raw + (0.002 if dataset != "oregon" else 0.0015)
                if dataset == "ena24" and budget == 0.4:
                    raw, calibrated = 0.708, 0.710
                for mode in self.protocol["modes"]:
                    rows.extend(
                        [
                            {"dataset": dataset, "role": "calibration", "method": "Raw-ActionValue-OQ", "mode": mode, "budget": budget, "official_ap50_95": raw},
                            {"dataset": dataset, "role": "calibration", "method": "PerAction-Isotonic-OQ", "mode": mode, "budget": budget, "official_ap50_95": calibrated},
                        ]
                    )
        gate = evaluate_gate(rows, self.protocol)
        self.assertTrue(gate["passed"])


if __name__ == "__main__":
    unittest.main()
