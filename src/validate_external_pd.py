"""
External validation: apply frozen PD pipelines (no retraining) to a separate PD cohort
with true H&Y labels. Welders are not valid external-validation targets (no true H&Y).

Expects the same PD sheet layout as training: columns resolvable to BBS, Mini-BEST, FES,
and H&Y (see load_pd_dataframe).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.utils import FEATURES, load_pd_dataframe, repo_root, write_json


def evaluate_frozen(
    data_path: Path,
    models_dir: Path,
    sheet: str | int,
    out_json: Path | None,
) -> dict:
    df = load_pd_dataframe(data_path, sheet=sheet)
    df = df.dropna(subset=list(FEATURES) + ["HY", "HY_bin"]).reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Need at least 2 rows with complete BBS, Mini, FES, and H&Y.")

    X = df[list(FEATURES)].values.astype(float)
    y_bin = df["HY_bin"].values.astype(int)
    y_mc = df["HY"].values.astype(int)

    bin_pack = joblib.load(models_dir / "hy_binary_pipeline.joblib")
    mc_pack = joblib.load(models_dir / "hy_multiclass_pipeline.joblib")

    y_bin_pred = bin_pack["pipeline"].predict(X)
    y_mc_pred = mc_pack["pipeline"].predict(X)

    stages = sorted(np.unique(np.concatenate([y_mc, y_mc_pred])))
    cm_mc = confusion_matrix(y_mc, y_mc_pred, labels=stages)

    result = {
        "n_rows": len(df),
        "sheet": str(sheet),
        "binary": {
            "accuracy": float(accuracy_score(y_bin, y_bin_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_bin, y_bin_pred)),
            "f1_macro": float(f1_score(y_bin, y_bin_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_bin, y_bin_pred).tolist(),
            "classification_report": classification_report(
                y_bin, y_bin_pred, target_names=["Early (I-II)", "Late (III-IV)"], zero_division=0
            ),
        },
        "multiclass": {
            "accuracy": float(accuracy_score(y_mc, y_mc_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_mc, y_mc_pred)),
            "f1_macro": float(f1_score(y_mc, y_mc_pred, average="macro", zero_division=0)),
            "within_one_stage_accuracy": float(np.mean(np.abs(y_mc - y_mc_pred) <= 1)),
            "confusion_matrix": cm_mc.tolist(),
            "stage_labels": [int(s) for s in stages],
            "classification_report": classification_report(
                y_mc,
                y_mc_pred,
                labels=stages,
                target_names=[f"Stage {int(s)}" for s in stages],
                zero_division=0,
            ),
        },
    }

    if out_json:
        write_json(out_json, result)
    return result


def main() -> None:
    root = repo_root()
    p = argparse.ArgumentParser(description="External PD validation with frozen pipelines.")
    p.add_argument("external_xlsx", type=Path, help="Excel file containing a PD sheet")
    p.add_argument(
        "--sheet",
        default="PD",
        help="Sheet name or index (default: PD)",
    )
    p.add_argument("--models-dir", type=Path, default=root / "models")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "outputs" / "metrics" / "external_validation.json",
    )
    args = p.parse_args()

    if not args.external_xlsx.is_file():
        print(f"Not found: {args.external_xlsx}", file=sys.stderr)
        sys.exit(1)
    if not (args.models_dir / "hy_multiclass_pipeline.joblib").is_file():
        print("Train first: python -m src.train_hy_model", file=sys.stderr)
        sys.exit(1)

    sheet: str | int = args.sheet
    try:
        sheet = int(sheet)
    except ValueError:
        pass

    ev = evaluate_frozen(args.external_xlsx, args.models_dir, sheet, args.output)
    print(f"External PD | n={ev['n_rows']} | binary acc={ev['binary']['accuracy']:.3f} | "
          f"multiclass acc={ev['multiclass']['accuracy']:.3f} | "
          f"within-1-stage={ev['multiclass']['within_one_stage_accuracy']:.3f}")
    print(f"Metrics -> {args.output}")


if __name__ == "__main__":
    main()
