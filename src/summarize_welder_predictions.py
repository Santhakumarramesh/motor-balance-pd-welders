"""Build a compact table: ID, PD-like stage, severity score, confidence (max stage probability)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import repo_root


def _stage_prob_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if re.match(r"^P_Stage\d+$", str(c))],
        key=lambda x: int(x.replace("P_Stage", "")),
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    pcols = _stage_prob_columns(df)
    if not pcols:
        raise ValueError("No P_Stage* probability columns found.")
    id_col = "ID" if "ID" in df.columns else df.columns[0]
    raw_id = df[id_col]
    fallback = pd.Series([f"row_{i + 1}" for i in range(len(df))], index=df.index, dtype=object)
    welder_id = raw_id.where(raw_id.notna(), fallback)
    out = pd.DataFrame(
        {
            "Welder_ID": welder_id,
            "Pred_HY_like_stage": df["Pred_Stage"],
            "PD_Severity_Score": df["PD_Severity_Score"],
            "Confidence": df[pcols].max(axis=1),
        }
    )
    return out


def main() -> None:
    root = repo_root()
    p = argparse.ArgumentParser(
        description="Summarize welder_predictions.xlsx to a small ID / stage / score / confidence table."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=root / "outputs" / "predictions" / "welder_predictions.xlsx",
        help="Path to welder_predictions.xlsx (after run_all or project_welders)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "outputs" / "predictions" / "welder_predictions_summary.csv",
        help="Write CSV (default: outputs/predictions/welder_predictions_summary.csv)",
    )
    p.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="Print table only; do not write CSV",
    )
    args = p.parse_args()

    if not args.input.is_file():
        print(f"Not found: {args.input} — run python run_all.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(args.input)
    out = summarize(df)

    if args.print_only:
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(out.to_string(index=False))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"Wrote {args.output} ({len(out)} rows)")


if __name__ == "__main__":
    main()
