from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
