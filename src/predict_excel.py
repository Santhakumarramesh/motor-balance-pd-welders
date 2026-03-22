"""Apply saved PD H&Y pipelines to any Excel with BBS, Mini-BEST, and FES columns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib

from src.utils import FEATURES, normalize_balance_columns, read_excel_sheet, repo_root


def run_predict(
    input_xlsx: Path,
    models_dir: Path,
    output_xlsx: Path,
    sheet: str | int | None = None,
    binary: bool = False,
) -> pd.DataFrame:
    raw = read_excel_sheet(Path(input_xlsx), sheet)
    feat = normalize_balance_columns(raw)
    mc_pack = joblib.load(models_dir / "hy_multiclass_pipeline.joblib")
    mc_pipe = mc_pack["pipeline"]
    mc_classes = np.array(mc_pack["classes"])

    X = feat[list(FEATURES)].values.astype(float)
    out = feat[["ID"]].copy()
    mc_proba = mc_pipe.predict_proba(X)
    for i, s in enumerate(mc_classes):
        out[f"P_Stage{int(s)}"] = mc_proba[:, i]
    out["Pred_Stage"] = mc_pipe.predict(X)
    out["PD_Severity_Score"] = sum(
        float(s) * mc_proba[:, i] for i, s in enumerate(mc_classes)
    )

    if binary:
        bin_pack = joblib.load(models_dir / "hy_binary_pipeline.joblib")
        out["Pred_Binary"] = bin_pack["pipeline"].predict(X)
        out["Pred_Binary_Label"] = out["Pred_Binary"].map(
            {0: "Early-like (I-II)", 1: "Late-like (III-IV)"}
        )

    out_xlsx = Path(output_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(out_xlsx, index=False)
    return out


def main() -> None:
    root = repo_root()
    p = argparse.ArgumentParser(
        description="Row-level PD-like H&Y predictions from an Excel file "
        "(requires BBS, Mini-BEST, FES columns; names auto-detected)."
    )
    p.add_argument("input_xlsx", type=Path, help="Path to .xlsx")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "outputs" / "predictions" / "inference_predictions.xlsx",
        help="Output Excel path",
    )
    p.add_argument(
        "--sheet",
        default=None,
        help="Sheet name or index (default: WD, else PD, else first sheet)",
    )
    p.add_argument("--models-dir", type=Path, default=root / "models")
    p.add_argument(
        "--binary",
        action="store_true",
        help="Also add binary Early/Late-like predictions",
    )
    args = p.parse_args()

    if not args.input_xlsx.is_file():
        print(f"Not found: {args.input_xlsx}", file=sys.stderr)
        sys.exit(1)
    if not (args.models_dir / "hy_multiclass_pipeline.joblib").is_file():
        print("Train models first: python -m src.train_hy_model", file=sys.stderr)
        sys.exit(1)

    sheet_arg: str | int | None = args.sheet
    if sheet_arg is not None:
        try:
            sheet_arg = int(sheet_arg)
        except ValueError:
            pass

    out = run_predict(args.input_xlsx, args.models_dir, args.output, sheet_arg, args.binary)
    print(f"Wrote {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
