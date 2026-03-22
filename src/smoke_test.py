"""Lightweight checks: data paths, PD/WD load, train artifacts, inference Excel."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import default_data_path, load_pd_dataframe, load_wd_dataframe, repo_root


def run_checks(root: Path, data: Path) -> None:
    assert data.is_file(), f"Data not found: {data}"

    pd_df = load_pd_dataframe(data)
    assert len(pd_df) > 0, "PD sheet empty"
    wd_df = load_wd_dataframe(data)
    assert len(wd_df) > 0, "WD sheet empty"

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        fig_dir = tdp / "figures"
        met_dir = tdp / "metrics"
        mod_dir = tdp / "models"
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "src.train_hy_model",
                "--data",
                str(data),
                "--out-fig",
                str(fig_dir),
                "--out-metrics",
                str(met_dir),
                "--models-dir",
                str(mod_dir),
            ],
            cwd=root,
        )
        assert (mod_dir / "hy_binary_pipeline.joblib").is_file()
        assert (mod_dir / "hy_multiclass_pipeline.joblib").is_file()
        assert (met_dir / "phase1_metrics.json").is_file()

        out_xlsx = tdp / "smoke_inference.xlsx"
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "src.predict_excel",
                str(data),
                "-o",
                str(out_xlsx),
                "--models-dir",
                str(mod_dir),
            ],
            cwd=root,
        )
        assert out_xlsx.is_file() and out_xlsx.stat().st_size > 0


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke test: load data, train, predict_excel.")
    p.add_argument("--data", type=Path, default=None, help="Excel path (default: data/...)")
    args = p.parse_args()
    root = repo_root()
    data = args.data or default_data_path()
    run_checks(root, data)
    print("smoke_test: OK")


if __name__ == "__main__":
    main()
