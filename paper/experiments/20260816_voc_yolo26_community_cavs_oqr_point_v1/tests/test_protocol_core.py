from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "voc_point_protocol_core", ROOT / "scripts/protocol_core.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("protocol_core.py is unavailable")
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


class ProtocolCoreTest(unittest.TestCase):
    def test_protocol_is_point_only_and_role_counts_are_exact(self) -> None:
        CORE.validate_protocol(PROTOCOL)
        self.assertEqual(PROTOCOL["bootstrap"]["replicates"], 0)
        self.assertEqual(PROTOCOL["roles"], {"fit": 15629, "calibration": 922, "final": 4952})

    def test_deterministic_split_is_disjoint_and_complete(self) -> None:
        identities = [f"2007:{index:06d}" for index in range(5011)]
        identities.extend(f"2012:{index:06d}" for index in range(11540))
        roles = CORE.deterministic_fit_calibration_split(
            identities,
            fit_count=15629,
            calibration_count=922,
            seed=20260816,
        )
        self.assertEqual(len(roles["fit"]), 15629)
        self.assertEqual(len(roles["calibration"]), 922)
        self.assertFalse(set(roles["fit"]) & set(roles["calibration"]))
        self.assertEqual(set(roles["fit"]) | set(roles["calibration"]), set(identities))

    def test_fold_membership_covers_each_identity_once(self) -> None:
        identities = [f"image:{index}" for index in range(101)]
        folds = CORE.deterministic_folds(identities, fold_count=5, seed=17)
        observed = [value for fold in folds for value in fold]
        self.assertEqual(len(observed), len(set(observed)))
        self.assertEqual(set(observed), set(identities))
        self.assertLessEqual(max(map(len, folds)) - min(map(len, folds)), 1)

    def test_fixed_byte_selection_never_exceeds_capacity(self) -> None:
        groups = [
            [
                {"image_id": 1, "action_id": "edge", "value": 0.0, "cost": 0.0},
                {"image_id": 1, "action_id": "original", "value": 3.0, "cost": 10.0},
                {"image_id": 1, "action_id": "q90", "value": 2.5, "cost": 6.0},
            ],
            [
                {"image_id": 2, "action_id": "edge", "value": 0.0, "cost": 0.0},
                {"image_id": 2, "action_id": "original", "value": 2.0, "cost": 10.0},
                {"image_id": 2, "action_id": "q90", "value": 1.5, "cost": 5.0},
            ],
        ]
        selected = CORE.select_actions_for_capacity(groups, capacity=10.0)
        self.assertLessEqual(sum(float(row["cost"]) for row in selected), 10.0)
        self.assertEqual(len({int(row["image_id"]) for row in selected}), len(selected))

    def test_checkpoint_names_must_match_voc_schema(self) -> None:
        expected = PROTOCOL["data"]["category_names"]
        CORE.validate_category_names({index: value for index, value in enumerate(expected)}, expected)
        with self.assertRaises(ValueError):
            CORE.validate_category_names({0: "person"}, expected)

    def test_training_plan_retrains_all_fifteen_full_fit_models(self) -> None:
        jobs = CORE.training_jobs(PROTOCOL)
        self.assertEqual(len(jobs), 15)
        self.assertEqual(len({job["id"] for job in jobs}), 15)
        self.assertTrue(all(job["role"] == "fit" for job in jobs))
        self.assertEqual(
            {job["method"] for job in jobs},
            {"CAVS-OQR-MAUR", "ORIC-AP5095", "MORIC-AP5095", "ORIC-AP50", "MORIC-AP50"},
        )

    def test_expected_point_matrix_covers_every_role_mode_budget_and_seed(self) -> None:
        expected = CORE.expected_point_keys(PROTOCOL)
        self.assertEqual(len(expected), 324)
        self.assertEqual(len(expected), len(set(expected)))
        observed = [
            {
                "role": role,
                "method": method,
                "seed": seed,
                "mode": mode,
                "budget": budget,
            }
            for role, method, seed, mode, budget in expected
        ]
        CORE.validate_point_rows(observed, PROTOCOL)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            CORE.validate_point_rows(observed[:-1], PROTOCOL)


if __name__ == "__main__":
    unittest.main()
