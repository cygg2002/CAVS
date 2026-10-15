from __future__ import annotations

from io import BytesIO
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from protocol import native_jpeg, q90_jpeg


class ProtocolTest(unittest.TestCase):
    def test_original_is_byte_exact_and_q90_is_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jpg"
            image = Image.new("RGB", (800, 400), (20, 80, 140))
            image.save(path, format="JPEG", quality=93)
            source = path.read_bytes()
            self.assertEqual(native_jpeg(path), source)
            first = q90_jpeg(path, long_side=640)
            second = q90_jpeg(path, long_side=640)
            self.assertEqual(first, second)
            with Image.open(BytesIO(first)) as encoded:
                self.assertEqual(encoded.size, (640, 320))

    def test_frozen_capacities_use_native_coco_jpegs(self) -> None:
        config = json.loads((ROOT / "configs" / "protocol.json").read_text())
        total = config["dataset"]["native_jpeg_total_bytes"]
        self.assertEqual(total, 814705164)
        self.assertEqual([total * x for x in config["budgets"]], [81470516.4, 203676291.0, 325882065.6])


if __name__ == "__main__":
    unittest.main()
