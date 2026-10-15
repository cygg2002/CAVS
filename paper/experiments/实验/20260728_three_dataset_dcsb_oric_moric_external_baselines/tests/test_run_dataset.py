from __future__ import annotations

import sys
from pathlib import Path
import json
import math
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_dataset import (  # noqa: E402
    layout_for_dataset,
    _load_reward_checkpoint,
    _role_features,
    _source_digest,
    _validate_reward_bundle,
    run_evaluate,
    run_preflight,
    run_rewards,
)
from runner_core import capture_hashes, file_sha256, role_source_paths  # noqa: E402


class DatasetRunnerTest(unittest.TestCase):
    def _write_reward_bundle(
        self, root: Path, *, source_digest: str = "source-digest"
    ) -> tuple[Path, Path, Path, Path]:
        root.mkdir(parents=True)
        protocol_path = root / "protocol.json"
        artifact_path = root / "fit_artifacts.npz"
        dcsb_path = root / "dcsb_model.json"
        completion_path = root / "completion.json"
        protocol_path.write_text('{"protocol_id":"test"}', encoding="utf-8")
        np.savez_compressed(
            artifact_path,
            source_digest=np.asarray(source_digest),
            rewards=np.asarray([0.1, 0.2]),
        )
        dcsb_path.write_text('{"count_threshold":1}', encoding="utf-8")
        completion_path.write_text(
            json.dumps(
                {
                    "complete": True,
                    "source_digest": source_digest,
                    "protocol_sha256": file_sha256(protocol_path),
                    "artifact_sha256": file_sha256(artifact_path),
                    "dcsb_model_sha256": file_sha256(dcsb_path),
                }
            ),
            encoding="utf-8",
        )
        return protocol_path, artifact_path, dcsb_path, completion_path

    def test_reward_bundle_rejects_changed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rewards"
            protocol_path, artifact_path, _, _ = self._write_reward_bundle(root)
            np.savez_compressed(
                artifact_path,
                source_digest=np.asarray("source-digest"),
                rewards=np.asarray([9.9, 0.2]),
            )
            with self.assertRaises(RuntimeError):
                _validate_reward_bundle(
                    root,
                    expected_source_digest="source-digest",
                    protocol_path=protocol_path,
                )

    def test_reward_bundle_rejects_changed_protocol_or_source_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rewards"
            protocol_path, _, _, _ = self._write_reward_bundle(root)
            with self.assertRaises(RuntimeError):
                _validate_reward_bundle(
                    root,
                    expected_source_digest="different-source",
                    protocol_path=protocol_path,
                )
            protocol_path.write_text('{"protocol_id":"changed"}', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _validate_reward_bundle(
                    root,
                    expected_source_digest="source-digest",
                    protocol_path=protocol_path,
                )

    def test_reward_checkpoint_rejects_inconsistent_array_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.npz"
            np.savez_compressed(
                checkpoint,
                source_digest=np.asarray("digest"),
                image_ids=np.asarray([1, 2]),
                rewards=np.asarray([0.1]),
                edge_maps=np.asarray([0.2, 0.3]),
                cloud_maps=np.asarray([0.4, 0.5]),
            )
            with self.assertRaises(RuntimeError):
                _load_reward_checkpoint(checkpoint, "digest", [1, 2, 3])

    def test_role_features_follow_sample_order_not_sorted_image_ids(self) -> None:
        data = {
            "samples": [
                {"image_id": 2, "width": 100, "height": 100},
                {"image_id": 1, "width": 100, "height": 100},
            ],
            "image_ids": [1, 2],
            "actions": {
                "edge": {
                    1: [{"category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.1}],
                    2: [{"category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}],
                }
            },
        }
        features = _role_features(data, [1], 1)
        self.assertAlmostEqual(features[0, 5], 0.9)
        self.assertAlmostEqual(features[1, 5], 0.1)

    def test_coco_and_oregon_use_their_frozen_canonical_roots(self) -> None:
        coco = layout_for_dataset("coco")
        oregon = layout_for_dataset("oregon")
        self.assertEqual(coco["common"].name, "common")
        self.assertEqual(coco["common"].parent.name, "canonical_v1")
        self.assertEqual(oregon["common"].parent.name, "canonical_oregon_event_v1")
        self.assertEqual(coco["output"], ROOT / "runs/coco")
        self.assertEqual(oregon["output"], ROOT / "runs/oregon")

    def test_unknown_dataset_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            layout_for_dataset("unknown")

    def test_preflight_validates_all_roles_and_writes_only_new_output(self) -> None:
        protocol = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
        protocol["expected_role_sizes"]["coco"] = {
            "fit": 2,
            "calibration": 2,
            "final": 2,
        }

        class FakeBenchmark:
            COMMON: Path | None = None

            @staticmethod
            def load_role(_dataset: str, role: str) -> dict:
                return {
                    "role": role,
                    "samples": [{"image_id": 1}, {"image_id": 2}],
                    "image_ids": [1, 2],
                }

        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            common = temp / "source/common"
            benchmark_root = temp / "source/benchmark/coco"
            output_root = temp / "new-runs"
            for role in ("fit", "calibration", "final"):
                role_root = common / "coco" / role
                role_root.mkdir(parents=True)
                actions = {}
                for action in ("edge", "original", "q90", "roi"):
                    path = role_root / f"{action}.json"
                    path.write_text("[]", encoding="utf-8")
                    actions[action] = str(path)
                (role_root / "samples.json").write_text("[]", encoding="utf-8")
                (role_root / "gt.json").write_text("{}", encoding="utf-8")
                (role_root / "source_manifest.json").write_text(
                    json.dumps({"action_sources": actions}), encoding="utf-8"
                )
            benchmark_root.mkdir(parents=True)
            (benchmark_root / "models.json").write_text(
                '{"action_heads":{"q90":{},"roi":{}}}', encoding="utf-8"
            )
            (benchmark_root / "action_gate.json").write_text(
                '{"chosen":{}}', encoding="utf-8"
            )
            report = run_preflight(
                "coco",
                protocol,
                layout={
                    "common": common,
                    "benchmark": benchmark_root,
                    "output": output_root / "coco",
                },
                benchmark=FakeBenchmark(),
                code_paths=(),
                output_root=output_root,
            )
            self.assertEqual(report["role_sizes"], {"fit": 2, "calibration": 2, "final": 2})
            self.assertTrue((output_root / "coco/preflight/completion.json").is_file())

    def test_rewards_use_fit_only_and_produce_resumable_artifact(self) -> None:
        protocol = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
        protocol["expected_role_sizes"]["coco"]["fit"] = 4
        protocol["oric"].update(
            {"context_size": 2, "checkpoint_interval": 2, "reward_workers": 1}
        )

        accessed_roles = []
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            common = temp / "source/common"
            role_root = common / "coco/fit"
            role_root.mkdir(parents=True)
            actions = {}
            for action in ("edge", "original", "q90", "roi"):
                path = role_root / f"{action}.json"
                path.write_text("[]", encoding="utf-8")
                actions[action] = str(path)
            gt = {
                "images": [
                    {"id": image_id, "width": 100, "height": 100}
                    for image_id in range(1, 5)
                ],
                "annotations": [
                    {"image_id": image_id, "category_id": 1, "bbox": [0, 0, 20, 20]}
                    for image_id in range(1, 5)
                ],
                "categories": [{"id": 1, "name": "object"}],
            }
            (role_root / "gt.json").write_text(json.dumps(gt), encoding="utf-8")
            (role_root / "samples.json").write_text("[]", encoding="utf-8")
            (role_root / "source_manifest.json").write_text(
                json.dumps({"action_sources": actions}), encoding="utf-8"
            )
            samples = [
                {"image_id": image_id, "width": 100, "height": 100}
                for image_id in range(1, 5)
            ]
            edge = {
                image_id: [
                    {
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [0, 0, 20, 20],
                        "score": 0.4 if image_id % 2 == 0 else 0.9,
                    }
                ]
                for image_id in range(1, 5)
            }
            original = {
                image_id: [
                    {
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [0, 0, 20, 20],
                        "score": 0.9,
                    }
                ]
                for image_id in range(1, 5)
            }

            class FakeBenchmark:
                COMMON: Path | None = None

                @staticmethod
                def load_role(_dataset: str, role: str) -> dict:
                    accessed_roles.append(role)
                    return {
                        "samples": samples,
                        "image_ids": [1, 2, 3, 4],
                        "gt_path": role_root / "gt.json",
                        "gt_map": {
                            image_id: [{"category_id": 1, "bbox": [0, 0, 20, 20]}]
                            for image_id in range(1, 5)
                        },
                        "actions": {"edge": edge, "original": original},
                    }

            output_root = temp / "new-runs"
            preflight = output_root / "coco/preflight"
            preflight.mkdir(parents=True)
            (preflight / "completion.json").write_text(
                '{"complete":true}', encoding="utf-8"
            )
            report = run_rewards(
                "coco",
                protocol,
                layout={
                    "common": common,
                    "benchmark": temp / "unused",
                    "output": output_root / "coco",
                },
                benchmark=FakeBenchmark(),
                code_paths=(),
                output_root=output_root,
            )
            self.assertEqual(accessed_roles, ["fit"])
            self.assertEqual(report["fit_images"], 4)
            artifact = output_root / "coco/rewards/fit_artifacts.npz"
            self.assertTrue(artifact.is_file())
            self.assertTrue((output_root / "coco/rewards/completion.json").is_file())

    def test_evaluate_reads_calibration_and_final_and_writes_65_rows(self) -> None:
        protocol = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
        protocol["expected_role_sizes"]["coco"].update(
            {"fit": 4, "calibration": 4, "final": 4}
        )
        accessed_roles = []

        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            common = temp / "source/common"
            benchmark_root = temp / "source/benchmark/coco"
            output_root = temp / "new-runs"
            output = output_root / "coco"
            fit_root = common / "coco/fit"
            fit_root.mkdir(parents=True)
            fit_actions = {}
            for action in ("edge", "original", "q90", "roi"):
                path = fit_root / f"{action}.json"
                path.write_text("[]", encoding="utf-8")
                fit_actions[action] = str(path)
            (fit_root / "gt.json").write_text("{}", encoding="utf-8")
            (fit_root / "samples.json").write_text("[]", encoding="utf-8")
            (fit_root / "source_manifest.json").write_text(
                json.dumps({"action_sources": fit_actions}), encoding="utf-8"
            )
            role_data = {}
            for role, first_id in (("calibration", 101), ("final", 201)):
                role_root = common / "coco" / role
                role_root.mkdir(parents=True)
                ids = list(range(first_id, first_id + 4))
                actions = {}
                for action in ("edge", "original", "q90", "roi"):
                    path = role_root / f"{action}.json"
                    path.write_text("[]", encoding="utf-8")
                    actions[action] = str(path)
                gt = {
                    "images": [
                        {"id": image_id, "width": 100, "height": 100}
                        for image_id in ids
                    ],
                    "annotations": [
                        {"image_id": image_id, "category_id": 1, "bbox": [0, 0, 20, 20]}
                        for image_id in ids
                    ],
                    "categories": [{"id": 1, "name": "object"}],
                }
                (role_root / "gt.json").write_text(json.dumps(gt), encoding="utf-8")
                (role_root / "samples.json").write_text("[]", encoding="utf-8")
                (role_root / "source_manifest.json").write_text(
                    json.dumps({"action_sources": actions}), encoding="utf-8"
                )
                samples = [
                    {
                        "image_id": image_id,
                        "width": 100,
                        "height": 100,
                        "payload_bytes": {"original": 1000, "q90": 600, "roi": 300},
                        "gains": {"original": 1.0, "q90": 0.8, "roi": 0.5},
                    }
                    for image_id in ids
                ]
                predictions = {
                    image_id: [
                        {
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [0, 0, 20, 20],
                            "score": 0.4 + 0.1 * index,
                        }
                    ]
                    for index, image_id in enumerate(ids)
                }
                role_data[role] = {
                    "dataset": "coco",
                    "role": role,
                    "samples": samples,
                    "sample_by_id": {sample["image_id"]: sample for sample in samples},
                    "image_ids": ids,
                    "gt_path": role_root / "gt.json",
                    "actions": {
                        "edge": predictions,
                        "original": predictions,
                        "q90": predictions,
                        "roi": predictions,
                    },
                }
            benchmark_root.mkdir(parents=True)
            (benchmark_root / "models.json").write_text(
                '{"action_heads":{"q90":{},"roi":{}},"m2e":{"weights":[]}}',
                encoding="utf-8",
            )
            (benchmark_root / "action_gate.json").write_text(
                '{"chosen":{}}', encoding="utf-8"
            )

            class FakeBenchmark:
                COMMON: Path | None = None

                @staticmethod
                def load_role(_dataset: str, role: str) -> dict:
                    accessed_roles.append(role)
                    return role_data[role]

                @staticmethod
                def chosen_actions(_heads: dict, _gate: dict, data: dict) -> dict[int, str]:
                    return {
                        sample["image_id"]: ("q90" if index % 2 else "original")
                        for index, sample in enumerate(data["samples"])
                    }

                @staticmethod
                def top_k_actions(data: dict, scores: np.ndarray, actions: dict, budget: float) -> dict:
                    count = int(math.floor(len(scores) * budget + 0.5))
                    order = np.argsort(-scores, kind="mergesort")[:count]
                    return {
                        data["samples"][index]["image_id"]: actions[data["samples"][index]["image_id"]]
                        for index in order
                    }

                @staticmethod
                def online_threshold(scores: np.ndarray, target: float) -> float:
                    count = max(1, int(round(len(scores) * target)))
                    return float(np.sort(scores)[::-1][count - 1])

                @staticmethod
                def calibrate_lambda(_cal: dict, _scores: np.ndarray, _actions: dict, _budget: float, _config: dict) -> dict:
                    return {"lambda": 0.0, "median_original_bytes": 1000.0}

                @staticmethod
                def audit_priority(_cal_scores: np.ndarray, final_scores: np.ndarray, _data: dict, _actions: dict, _lambda: float, _median: float) -> np.ndarray:
                    return final_scores

                @staticmethod
                def byte_selection(data: dict, priorities: np.ndarray, actions: dict, budget: float) -> dict:
                    return FakeBenchmark.top_k_actions(data, priorities, actions, budget)

                @staticmethod
                def evaluate(method: str, mode: str, budget: float, data: dict, actions: dict, _models: dict) -> dict:
                    request = sum(
                        data["sample_by_id"][image_id]["payload_bytes"][action]
                        for image_id, action in actions.items()
                    )
                    return {
                        "dataset": data["dataset"],
                        "role": data["role"],
                        "method": method,
                        "mode": mode,
                        "nominal_budget": budget,
                        "selected_images": len(actions),
                        "request_kb_per_image": request / len(data["samples"]) / 1024.0,
                        "official_map50_95": 0.5,
                    }

            preflight = output / "preflight"
            preflight.mkdir(parents=True)
            (preflight / "completion.json").write_text(
                '{"complete":true}', encoding="utf-8"
            )
            rewards_root = output / "rewards"
            rewards_root.mkdir(parents=True)
            reward_source_paths = [ROOT / "configs/protocol.json"]
            reward_source_paths.extend(role_source_paths(common, "coco", "fit"))
            reward_source_digest = _source_digest(capture_hashes(reward_source_paths))
            np.savez_compressed(
                rewards_root / "fit_artifacts.npz",
                source_digest=np.asarray(reward_source_digest),
                image_ids=np.arange(1, 5),
                category_ids=np.asarray([1]),
                features=np.arange(4 * 126, dtype=float).reshape(4, 126),
                rewards=np.asarray([-1.0, 0.0, 1.0, 2.0]),
            )
            (rewards_root / "dcsb_model.json").write_text(
                json.dumps(
                    {
                        "confidence_threshold": 0.2,
                        "count_threshold": 1,
                        "area_threshold": 0.1,
                        "detector_confidence": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            (rewards_root / "completion.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "source_digest": reward_source_digest,
                        "protocol_sha256": file_sha256(ROOT / "configs/protocol.json"),
                        "artifact_sha256": file_sha256(
                            rewards_root / "fit_artifacts.npz"
                        ),
                        "dcsb_model_sha256": file_sha256(
                            rewards_root / "dcsb_model.json"
                        ),
                    }
                ),
                encoding="utf-8",
            )

            def fake_fit(features: np.ndarray, targets: np.ndarray, _weights: np.ndarray, _config: dict) -> dict:
                return {"offset": float(targets.mean()), "input_dim": features.shape[1]}

            def fake_predict(checkpoint: dict, features: np.ndarray) -> np.ndarray:
                return features[:, 0] + checkpoint["offset"]

            report = run_evaluate(
                "coco",
                protocol,
                layout={
                    "common": common,
                    "benchmark": benchmark_root,
                    "output": output,
                },
                benchmark=FakeBenchmark(),
                code_paths=(),
                output_root=output_root,
                fit_model=fake_fit,
                predict_model=fake_predict,
            )
            self.assertEqual(accessed_roles, ["calibration", "final"])
            self.assertEqual(report["rows"], 65)
            with (output / "evaluation/results.csv").open(encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in handle) - 1, 65)


if __name__ == "__main__":
    unittest.main()
