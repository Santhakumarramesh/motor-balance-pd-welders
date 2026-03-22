"""PD vs welder group discrimination (5-fold CV); exploratory, not the H&Y model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train_hy_model import FEATURE_SETS, MODEL_NAMES, make_pipeline
from src.utils import default_data_path, load_pd_dataframe, load_wd_dataframe, repo_root, write_json


def build_group_table(data_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    pd_df = load_pd_dataframe(data_path)
    wd_df = load_wd_dataframe(data_path)
    rows = []
    for _, r in pd_df.iterrows():
        rows.append(
            {"BBS": r["BBS"], "Mini": r["Mini"], "FES": r["FES"], "Group": 0, "Label": "PD"}
        )
    for _, r in wd_df.iterrows():
        rows.append(
            {"BBS": r["BBS"], "Mini": r["Mini"], "FES": r["FES"], "Group": 1, "Label": "Welder"}
        )
    df = pd.DataFrame(rows).dropna(subset=["BBS", "Mini", "FES", "Group"])
    y = df["Group"].values.astype(int)
    return df, y


def run_benchmark(data_path: Path, out_metrics: Path, out_fig: Path | None) -> dict:
    df, y = build_group_table(data_path)
    n_pd = int((y == 0).sum())
    n_wd = int((y == 1).sum())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results: dict = {
        "task": "pd_vs_welder_group",
        "n_total": len(y),
        "n_pd": n_pd,
        "n_welder": n_wd,
        "validation": "5-fold_stratified_cv",
        "feature_sets": {},
    }

    plot_rows = []

    for fs_name, cols in FEATURE_SETS.items():
        X = df[list(cols)].values.astype(float)
        fs_out: dict = {}
        for mname in MODEL_NAMES:
            pipe = make_pipeline(mname)
            y_pred = cross_val_predict(pipe, X, y, cv=cv)
            acc = accuracy_score(y, y_pred)
            bal = balanced_accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average="macro", zero_division=0)
            fs_out[mname] = {
                "accuracy": acc,
                "balanced_accuracy": bal,
                "f1_macro": f1,
            }
            plot_rows.append(
                {"feature_set": fs_name, "model": mname, "accuracy": acc, "f1_macro": f1}
            )
        results["feature_sets"][fs_name] = fs_out

    write_json(out_metrics / "group_discrimination.json", results)

    if out_fig is not None:
        out_fig.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        pr = pd.DataFrame(plot_rows)
        for ax, metric, title in zip(
            axes,
            ["accuracy", "f1_macro"],
            ["Accuracy (5-fold CV)", "Macro F1 (5-fold CV)"],
        ):
            pivot = pr.pivot(index="feature_set", columns="model", values=metric)
            pivot.plot(kind="bar", ax=ax, rot=0, width=0.85)
            ax.set_title(title)
            ax.set_xlabel("Feature set")
            ax.legend(title="Model", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        plt.suptitle(
            "Supporting analysis: PD vs Welder (group discrimination; age-confounded)",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(out_fig / "fig_08_group_discrimination.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(
        f"Group benchmark | n={len(y)} (PD={n_pd}, Welder={n_wd}) | -> {out_metrics}/group_discrimination.json"
    )
    return results


def main() -> None:
    root = repo_root()
    p = argparse.ArgumentParser(description="PD vs Welder group discrimination (5-fold CV).")
    p.add_argument("--data", type=Path, default=default_data_path())
    p.add_argument("--out-metrics", type=Path, default=root / "outputs" / "metrics")
    p.add_argument("--out-fig", type=Path, default=root / "outputs" / "figures")
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args()
    if not args.data.is_file():
        print(f"Data not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    run_benchmark(args.data, args.out_metrics, None if args.no_figure else args.out_fig)


if __name__ == "__main__":
    main()
