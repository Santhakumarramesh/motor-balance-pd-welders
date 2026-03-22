#!/usr/bin/env python3
"""Run Phase 1 training, benchmark, Phase 2 projection, paper figures, summary report."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Full analysis pipeline from repo root.")
    parser.add_argument(
        "--exposure",
        action="store_true",
        help="Phase 2: include exploratory exposure correlations and third fig_07 panel",
    )
    args = parser.parse_args()

    subprocess.check_call([sys.executable, "-m", "src.train_hy_model"], cwd=ROOT)
    subprocess.check_call([sys.executable, "-m", "src.benchmark_group_discrimination"], cwd=ROOT)
    proj = [sys.executable, "-m", "src.project_welders"]
    if args.exposure:
        proj.append("--exposure")
    subprocess.check_call(proj, cwd=ROOT)
    subprocess.check_call([sys.executable, "-m", "src.generate_paper_figures"], cwd=ROOT)
    subprocess.check_call([sys.executable, "-m", "src.write_summary_report"], cwd=ROOT)


if __name__ == "__main__":
    main()
