#!/usr/bin/env python3
"""Compatibility entry point for the dependency-free SVG figure builder."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("build_system_figures.cjs")


def main() -> None:
    node = os.environ.get("NODE", "node")
    subprocess.run([node, str(SCRIPT)], check=True)


if __name__ == "__main__":
    main()
