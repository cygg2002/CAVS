from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/prepare_data.py"
SPEC = importlib.util.spec_from_file_location("voc_gluoncv_prepare", PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(PATH)
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


class PrepareDataTest(unittest.TestCase):
    def test_parse_annotation_retains_difficult_objects(self) -> None:
        payload = """<annotation>
          <size><width>100</width><height>80</height></size>
          <object><name>cat</name><difficult>0</difficult>
            <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>20</ymax></bndbox>
          </object>
          <object><name>cat</name><difficult>1</difficult>
            <bndbox><xmin>20</xmin><ymin>30</ymin><xmax>40</xmax><ymax>50</ymax></bndbox>
          </object>
        </annotation>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.xml"
            path.write_text(payload, encoding="utf-8")
            parsed = PREPARE.parse_annotation(path, {"cat": 1})
        self.assertEqual(len(parsed["objects"]), 2)
        self.assertEqual([row["difficult"] for row in parsed["objects"]], [False, True])
        self.assertEqual(parsed["objects"][0]["bbox"], [0.0, 1.0, 10.0, 19.0])

    def test_frozen_role_import_rejects_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            role_root = source / "runs/data/roles/fit"
            role_root.mkdir(parents=True)
            image = source / "sample.jpg"
            image.write_bytes(b"image")
            gt = {
                "images": [
                    {
                        "id": 1,
                        "identity": "2007:000001",
                        "absolute_path": str(image),
                        "original_bytes": 5,
                    }
                ],
                "annotations": [],
                "categories": [],
            }
            gt_path = role_root / "gt.json"
            gt_path.write_text(json.dumps(gt), encoding="utf-8")
            images_path = role_root / "images.txt"
            images_path.write_text(f"{image}\n", encoding="utf-8")
            contract = {
                "gt_sha256": PREPARE.DATA.file_sha256(gt_path),
                "images_sha256": PREPARE.DATA.file_sha256(images_path),
                "identity_sha256": hashlib.sha256(b"2007:000001").hexdigest(),
            }
            original_count = PREPARE.PROTOCOL["roles"]["fit"]
            PREPARE.PROTOCOL["roles"]["fit"] = 1
            try:
                report = PREPARE.validate_source_role(source, "fit", contract)
                self.assertEqual(report["images"], 1)
                contract["gt_sha256"] = "0" * 64
                with self.assertRaisesRegex(RuntimeError, "ground-truth hash changed"):
                    PREPARE.validate_source_role(source, "fit", contract)
            finally:
                PREPARE.PROTOCOL["roles"]["fit"] = original_count


if __name__ == "__main__":
    unittest.main()
