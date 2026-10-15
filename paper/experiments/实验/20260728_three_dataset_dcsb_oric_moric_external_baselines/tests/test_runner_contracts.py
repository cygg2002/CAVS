from __future__ import annotations

import sys
from pathlib import Path
import csv
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from runner_core import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    atomic_write_pickle,
    atomic_write_rows,
    assert_hashes_unchanged,
    assert_output_path,
    capture_hashes,
    role_source_paths,
    roles_for_stage,
    validate_role_data,
)


class RunnerContractTest(unittest.TestCase):
    def test_output_path_must_remain_below_this_experiment_runs(self) -> None:
        allowed = ROOT / "runs/coco/results.csv"
        self.assertEqual(assert_output_path(allowed, ROOT / "runs"), allowed.resolve())
        with self.assertRaises(RuntimeError):
            assert_output_path(ROOT.parent / "completed/results.csv", ROOT / "runs")

    def test_source_hash_guard_detects_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.json"
            source.write_text("before", encoding="utf-8")
            hashes = capture_hashes([source])
            assert_hashes_unchanged(hashes)
            source.write_text("after", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                assert_hashes_unchanged(hashes)

    def test_atomic_json_write_is_confined_to_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "runs"
            target = output / "unit/report.json"
            atomic_write_json(target, {"complete": True}, output)
            self.assertEqual(target.read_text(encoding="utf-8"), '{\n  "complete": true\n}\n')
            with self.assertRaises(RuntimeError):
                atomic_write_json(Path(temporary) / "outside.json", {}, output)

    def test_atomic_npz_write_round_trips_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "runs"
            target = output / "unit/features.npz"
            atomic_write_npz(target, output, values=np.asarray([1.0, 2.0]))
            with np.load(target) as artifact:
                np.testing.assert_allclose(artifact["values"], [1.0, 2.0])

    def test_atomic_csv_and_pickle_writes_are_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "runs"
            csv_path = output / "unit/results.csv"
            pickle_path = output / "unit/model.pkl"
            atomic_write_rows(csv_path, [{"method": "DCSB", "ap": 0.5}], output)
            atomic_write_pickle(pickle_path, {"epoch": 3}, output)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(list(csv.DictReader(handle))[0]["method"], "DCSB")
            import pickle

            with pickle_path.open("rb") as handle:
                self.assertEqual(pickle.load(handle), {"epoch": 3})

    def test_role_contract_checks_count_and_unique_image_ids(self) -> None:
        data = {"samples": [{"image_id": 1}, {"image_id": 2}], "image_ids": [1, 2]}
        validate_role_data(data, dataset="unit", role="fit", expected_count=2)
        with self.assertRaises(RuntimeError):
            validate_role_data(data, dataset="unit", role="fit", expected_count=3)
        duplicate = {
            "samples": [{"image_id": 1}, {"image_id": 1}],
            "image_ids": [1],
        }
        with self.assertRaises(RuntimeError):
            validate_role_data(duplicate, dataset="unit", role="fit", expected_count=2)

    def test_stage_roles_keep_final_out_of_reward_generation(self) -> None:
        self.assertEqual(roles_for_stage("rewards"), ("fit",))
        self.assertEqual(roles_for_stage("evaluate"), ("calibration", "final"))
        self.assertEqual(roles_for_stage("preflight"), ("fit", "calibration", "final"))

    def test_role_source_paths_include_every_manifest_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            common = Path(temporary)
            role_root = common / "unit/fit"
            role_root.mkdir(parents=True)
            edge = common / "edge.json"
            original = common / "original.json"
            for path in (edge, original, role_root / "samples.json", role_root / "gt.json"):
                path.write_text("[]", encoding="utf-8")
            (role_root / "source_manifest.json").write_text(
                '{"action_sources":{"edge":"%s","original":"%s"}}'
                % (edge, original),
                encoding="utf-8",
            )
            observed = set(role_source_paths(common, "unit", "fit"))
            self.assertEqual(
                observed,
                {
                    role_root / "source_manifest.json",
                    role_root / "samples.json",
                    role_root / "gt.json",
                    edge,
                    original,
                },
            )


if __name__ == "__main__":
    unittest.main()
