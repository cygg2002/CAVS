from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from online_control import CausalByteController
from run_contract import (
    PROTOCOL_ID,
    append_committed_frame,
    build_run_contract,
    ensure_run_contract,
    exclusive_run_lock,
    restore_committed_prefix,
    verify_rknn_helpers_manifest,
)


class RunContractTest(unittest.TestCase):
    def _artifact(self, root: Path, name: str, content: bytes) -> Path:
        path = root / name
        path.write_bytes(content)
        return path

    def _contract(self, root: Path, *, run_id: str = "formal-cavs-10") -> dict:
        sequence = self._artifact(root, "sequence.json", b'{"records":[]}\n')
        policy = self._artifact(root, "policy.json", b'{"method":"cavs_oqr"}\n')
        control = self._artifact(root, "control.json", b'{"price":0.1}\n')
        rknn = self._artifact(root, "edge.rknn", b"rknn")
        helpers = self._artifact(root, "rknn_helpers.json", b'{"schema":"helpers"}\n')
        return build_run_contract(
            run_id=run_id,
            method="cavs_oqr",
            budget="0.1",
            sequence_path=sequence,
            policy_path=policy,
            control_path=control,
            rknn_path=rknn,
            cloud_sha256="c" * 64,
            action_ids=("edge", "original", "q90", "current@640"),
            random_seed=None,
            rknn_helpers_manifest_path=helpers,
        )

    def test_contract_binds_every_runtime_artifact_and_is_immutable(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self._contract(root)
            run_dir = root / "runs" / contract["run_id"]
            digest = ensure_run_contract(run_dir, contract)

            self.assertEqual(len(digest), 64)
            self.assertEqual(contract["sequence_sha256"], contract["artifacts"]["sequence"]["sha256"])
            self.assertEqual(contract["policy_sha256"], contract["artifacts"]["policy"]["sha256"])
            self.assertEqual(contract["control_sha256"], contract["artifacts"]["control"]["sha256"])
            self.assertEqual(contract["rknn_sha256"], contract["artifacts"]["rknn"]["sha256"])
            self.assertEqual(
                contract["rknn_helpers_manifest_sha256"],
                contract["artifacts"]["rknn_helpers"]["sha256"],
            )
            self.assertEqual(contract["cloud_sha256"], "c" * 64)
            self.assertIsNone(contract["random_seed"])
            self.assertTrue(contract["zero_lookahead"])
            self.assertFalse(contract["global_plan"])
            self.assertEqual(contract["buffer_frames"], 0)
            self.assertEqual(PROTOCOL_ID, "coco_rknn_fp16_causal_online_hil_final_v2")
            self.assertEqual(contract["protocol_id"], PROTOCOL_ID)
            self.assertNotIn("token", json.dumps(contract).lower())

            self.assertEqual(ensure_run_contract(run_dir, contract), digest)
            changed = {**contract, "cloud_sha256": "d" * 64}
            with self.assertRaisesRegex(RuntimeError, "run ID contract mismatch"):
                ensure_run_contract(run_dir, changed)

    def test_contract_rejects_unsafe_run_id_and_unbound_hash(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "run ID"):
                self._contract(root, run_id="../escape")
            with self.assertRaisesRegex(ValueError, "cloud SHA-256"):
                build_run_contract(
                    run_id="formal",
                    method="confidence",
                    budget="0.1",
                    sequence_path=self._artifact(root, "sequence.json", b"{}"),
                    policy_path=self._artifact(root, "method.json", b"{}"),
                    control_path=self._artifact(root, "control.json", b"{}"),
                    rknn_path=self._artifact(root, "edge.rknn", b"x"),
                    cloud_sha256="not-a-hash",
                    action_ids=("edge", "original"),
                    rknn_helpers_manifest_path=self._artifact(
                        root, "helpers.json", b"{}"
                    ),
                )

    def test_random_seed_is_bound_and_run_lock_has_one_writer(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            contract = build_run_contract(
                run_id="random-10", method="random", budget="0.1",
                sequence_path=self._artifact(root, "sequence.json", b"{}"),
                policy_path=self._artifact(root, "policy.json", b"{}"),
                control_path=self._artifact(root, "control.json", b"{}"),
                rknn_path=self._artifact(root, "edge.rknn", b"x"),
                cloud_sha256="c" * 64,
                action_ids=("edge", "original"), random_seed=20260720,
                rknn_helpers_manifest_path=self._artifact(
                    root, "helpers.json", b"{}"
                ),
            )
            self.assertEqual(contract["random_seed"], 20260720)

            run_dir = root / "runs" / "random-10"
            with exclusive_run_lock(run_dir):
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with exclusive_run_lock(run_dir):
                        pass

    def test_contract_rejects_disagreeing_artifact_hash_alias(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self._contract(root)
            contract["artifacts"]["policy"]["sha256"] = "d" * 64
            with self.assertRaisesRegex(ValueError, "artifact hash"):
                ensure_run_contract(root / "run", contract)

    def test_helpers_manifest_binds_the_external_rknn_modules(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            helper_root = root / "helpers"
            helper_root.mkdir()
            entries = []
            for name, content in (
                ("coco_val_rknn.py", b"preprocess-v1"),
                ("yolo26_decode.py", b"decode-v1"),
            ):
                path = helper_root / name
                path.write_bytes(content)
                entries.append({
                    "path": name,
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                })
            manifest = root / "helpers.json"
            manifest.write_text(
                json.dumps({
                    "schema": "cavs_rknn_helpers_manifest_v1",
                    "files": entries,
                }),
                encoding="utf-8",
            )
            verified = verify_rknn_helpers_manifest(manifest, helper_root)
            self.assertEqual(verified["files"], 2)
            (helper_root / "yolo26_decode.py").write_bytes(b"decode-v2")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_rknn_helpers_manifest(manifest, helper_root)

    def test_resume_replays_every_committed_frame_and_restores_state(self) -> None:
        sequence = [
            {"image_id": 1, "file_name": "0001.jpg", "width": 100, "height": 80},
            {"image_id": 2, "file_name": "0002.jpg", "width": 100, "height": 80},
        ]
        fresh = CausalByteController(
            budget="0.4",
            price=0.0,
            median_original_bytes=100,
            action_ids=("edge", "original", "q90"),
        )
        payloads = {"edge": 0, "original": 100, "q90": 25}
        scores = {"edge": 0.0, "original": 2.0, "q90": 1.0}
        decision = fresh.decide(1, payloads, scores)
        record = {
            "position": 0,
            "image_id": 1,
            "file_name": "0001.jpg",
            "decision": decision.__dict__,
            "controller_state": fresh.snapshot(),
            "payload_bytes": payloads,
            "scores": scores,
            "resolved_boxes": [],
        }
        with TemporaryDirectory() as directory:
            frames = Path(directory) / "frames.jsonl"
            append_committed_frame(
                frames,
                record,
                run_id="formal-cavs-10",
                contract_sha256="a" * 64,
            )
            restored, position, rows = restore_committed_prefix(
                frames,
                sequence,
                contract_sha256="a" * 64,
                controller=CausalByteController(
                    budget="0.4",
                    price=0.0,
                    median_original_bytes=100,
                    action_ids=("edge", "original", "q90"),
                ),
            )

            self.assertEqual(position, 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(restored.snapshot(), fresh.snapshot())
            saved = json.loads(frames.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "committed")

    def test_resume_rejects_non_prefix_tampering_and_partial_json(self) -> None:
        sequence = [{"image_id": 1, "file_name": "0001.jpg", "width": 100, "height": 80}]
        controller = CausalByteController(
            budget="0.4", price=0.0, median_original_bytes=100,
            action_ids=("edge", "original"),
        )
        with TemporaryDirectory() as directory:
            frames = Path(directory) / "frames.jsonl"
            frames.write_text('{"status":"committed"', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid frames.jsonl"):
                restore_committed_prefix(
                    frames,
                    sequence,
                    contract_sha256="a" * 64,
                    controller=controller,
                )

            frames.write_text(
                json.dumps({
                    "schema_version": 1, "status": "committed", "run_id": "r",
                    "contract_sha256": "a" * 64,
                    "position": 0, "image_id": 999, "file_name": "0001.jpg",
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "sequence prefix mismatch"):
                restore_committed_prefix(
                    frames,
                    sequence,
                    contract_sha256="a" * 64,
                    controller=controller,
                )


if __name__ == "__main__":
    unittest.main()
