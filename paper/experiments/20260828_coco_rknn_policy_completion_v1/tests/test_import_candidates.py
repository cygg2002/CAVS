from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "rknn_import_candidates", SCRIPTS / "import_candidates.py"
)
assert SPEC and SPEC.loader
IMPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORT)


class ImportCandidatePathTests(unittest.TestCase):
    def test_resolves_actual_usb_layout(self) -> None:
        board_root = Path("/mnt/usb/rk3588")

        input_path, candidate_root = IMPORT.board_role_paths(
            board_root, backend="fp16", role="fit"
        )

        self.assertEqual(
            input_path,
            board_root / "deployment_aligned_inputs/manifests/fit.json",
        )
        self.assertEqual(
            candidate_root,
            board_root
            / "cavs_oqr_edge_cloud/runs/deployment_aligned/fp16/fit",
        )


if __name__ == "__main__":
    unittest.main()
