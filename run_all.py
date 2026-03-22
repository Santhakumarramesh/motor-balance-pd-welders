#!/usr/bin/env python3
"""Run Phase 1 training then Phase 2 welder projection (from repo root)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    subprocess.check_call([sys.executable, "-m", "src.train_hy_model"], cwd=ROOT)
    subprocess.check_call([sys.executable, "-m", "src.project_welders"], cwd=ROOT)


if __name__ == "__main__":
    main()
