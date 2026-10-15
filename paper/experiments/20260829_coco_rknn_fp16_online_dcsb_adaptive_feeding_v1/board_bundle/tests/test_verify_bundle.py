from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from verify_bundle import build_manifest, verify_manifest


class VerifyBundleTest(unittest.TestCase):
    def test_verifies_assets_and_rejects_tampering(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            asset = root / "scripts" / "runner.py"
            asset.write_text("print('ok')\n", encoding="utf-8")
            manifest_path = root / "BUNDLE_MANIFEST.json"
            manifest_path.write_text(
                json.dumps(build_manifest(root, manifest_path)), encoding="utf-8"
            )

            self.assertEqual(verify_manifest(root, manifest_path)["assets"], 1)
            asset.write_text("print('changed')\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_manifest(root, manifest_path)


if __name__ == "__main__":
    unittest.main()
