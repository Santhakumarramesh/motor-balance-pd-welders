"""Welder projection: apply fitted PD H&Y pipelines; export tables and figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib

from src.utils import (
    FEATURES,
    default_data_path,
    load_pd_dataframe,
    load_wd_dataframe,
    repo_root,
    validate_ranges,
    w_stage,
    write_json,
)


def compare_groups_figure(pd_df: pd.DataFrame, wd_df: pd.DataFrame, out_path: Path):
    rows = []
    for _, r in pd_df.iterrows():
        rows.append(
            {
                "Group": "PD",
                "BBS": r["BBS"],
                "Mini": r["Mini"],
                "FES": r["FES"],
            }
        )
    for _, r in wd_df.iterrows():
        rows.append(
            {
                "Group": "Welder",
                "BBS": r["BBS"],
                "Mini": r["Mini"],
                "FES": r["FES"],
            }
        )
    long = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title in zip(
        axes,
        FEATURES,
        ["BBS (0–56, higher=better)", "Mini-BEST (0–28, higher=better)", "FES (higher=worse)"],
    ):
        sub = long.dropna(subset=[col])
        sns.boxplot(
            data=sub,
            x="Group",
            y=col,
            ax=ax,
            hue="Group",
            palette={"PD": "#3498db", "Welder": "#e67e22"},
            dodge=False,
            legend=False,
        )
        sns.stripplot(
            data=sub, x="Group", y=col, ax=ax, color="k", alpha=0.5, size=4, jitter=0.15
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Balance scales: PD vs Welders", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_projection(
    data_path: Path,
    models_dir: Path,
    out_fig: Path,
    out_pred: Path,
    *,
    exposure: bool = False,
):
    bin_pack = joblib.load(models_dir / "hy_binary_pipeline.joblib")
    mc_pack = joblib.load(models_dir / "hy_multiclass_pipeline.joblib")

    pd_df = load_pd_dataframe(data_path)
    wd_df = load_wd_dataframe(data_path)
    for w in validate_ranges(wd_df, "WD"):
        print("Warning:", w)

    out_fig.mkdir(parents=True, exist_ok=True)
    out_pred.mkdir(parents=True, exist_ok=True)

    compare_groups_figure(pd_df, wd_df, out_fig / "fig_06_pd_vs_welder_balance.png")

    wd_clean = wd_df.dropna(subset=list(FEATURES)).copy()
    if len(wd_clean) == 0:
        print("No welders with complete BBS/Mini/FES — aborting.", file=sys.stderr)
        sys.exit(2)

    X = wd_clean[list(FEATURES)].values.astype(float)

    bin_pipe = bin_pack["pipeline"]
    mc_pipe = mc_pack["pipeline"]
    mc_classes = np.array(mc_pack["classes"])

    wd_clean["Pred_Binary"] = bin_pipe.predict(X)
    wd_clean["Pred_Binary_Lbl"] = wd_clean["Pred_Binary"].map(
        {0: "Early-like (I-II)", 1: "Late-like (III-IV)"}
    )

    mc_proba = mc_pipe.predict_proba(X)
    for i, s in enumerate(mc_classes):
        wd_clean[f"P_Stage{int(s)}"] = mc_proba[:, i]

    wd_clean["Pred_Stage"] = mc_pipe.predict(X)
    wd_clean["PD_Severity_Score"] = sum(
        float(s) * mc_proba[:, i] for i, s in enumerate(mc_classes)
    )
    trained_max = float(np.max(mc_classes))
    wd_clean["At_Trained_Upper_Boundary"] = (
        wd_clean["Pred_Stage"].astype(float) == trained_max
    ).astype(int)
    wd_clean["Trained_HY_max_stage"] = int(trained_max)
    _interp_upper = (
        "Upper-bound (Stage IV-like ceiling; beyond-range possible)"
    )
    _interp_within = "Within trained PD severity range"
    wd_clean["Interpretation"] = np.where(
        wd_clean["At_Trained_Upper_Boundary"] == 1,
        _interp_upper,
        _interp_within,
    )

    if "WeldYrs" in wd_clean.columns:
        wd_clean["W_Stage"] = wd_clean["WeldYrs"].apply(w_stage)

    if exposure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax0, ax1, ax2 = axes[0], axes[1], axes[2]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        ax0, ax1 = axes[0], axes[1]
        ax2 = None

    ax0.hist(wd_clean["PD_Severity_Score"], bins=8, color="steelblue", edgecolor="k", alpha=0.8)
    ax0.axvline(
        wd_clean["PD_Severity_Score"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean={wd_clean['PD_Severity_Score'].mean():.2f}",
    )
    ax0.set_xlabel("PD-like severity score (1–4)")
    ax0.set_ylabel("Welders")
    ax0.set_title("Welder distribution in\nPD severity space")
    ax0.legend()
    ax0.grid(alpha=0.3)

    colors_stage = {1: "#3498db", 2: "#2ecc71", 3: "#f39c12", 4: "#e74c3c"}
    bottoms = np.zeros(len(wd_clean))
    for s in mc_classes:
        col = f"P_Stage{int(s)}"
        if col in wd_clean.columns:
            ax1.bar(
                range(len(wd_clean)),
                wd_clean[col],
                bottom=bottoms,
                label=f"Stage {int(s)}",
                color=colors_stage.get(int(s), "#888"),
                alpha=0.85,
            )
            bottoms += wd_clean[col].values
    ax1.set_xticks(range(len(wd_clean)))
    ax1.set_xticklabels(wd_clean["ID"].values, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Probability")
    ax1.set_title("Stage probability per welder")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    if exposure and ax2 is not None:
        sub = wd_clean.dropna(subset=["WeldYrs", "PD_Severity_Score"])
        if len(sub) >= 3:
            ax2.scatter(
                sub["WeldYrs"],
                sub["PD_Severity_Score"],
                c=sub.get("W_Stage", 1),
                cmap="YlOrRd",
                s=80,
                edgecolors="k",
                linewidths=0.6,
                alpha=0.9,
            )
            m, b, *_ = stats.linregress(sub["WeldYrs"], sub["PD_Severity_Score"])
            xl = np.linspace(sub["WeldYrs"].min(), sub["WeldYrs"].max(), 100)
            rho, pval = stats.spearmanr(sub["WeldYrs"], sub["PD_Severity_Score"])
            ax2.plot(xl, m * xl + b, "r--", linewidth=1.8)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax2.set_title(f"Severity score vs welding years\nρ={rho:+.3f} ({sig})")
        ax2.set_xlabel("Total years in welding")
        ax2.set_ylabel("PD-like severity score")
        ax2.grid(alpha=0.3)

    plt.suptitle(
        "Welder projection into PD motor-balance severity space\n"
        "(PD-referenced; not a clinical diagnosis)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_fig / "fig_07_welder_projection.png", dpi=150, bbox_inches="tight")
    plt.close()

    metrics_dir = out_pred.parent / "metrics"
    if exposure:
        exposure_vars = [
            ("WeldYrs", "Total years in welding"),
            ("HrsPerDay", "Hours per day"),
            ("FumeExp", "Fume exposure"),
            ("VibExp", "Vibration exposure"),
            ("NoiseExp", "Noise exposure"),
            ("RespPPE", "Respiratory PPE"),
            ("FallHist", "History of falls"),
            ("W_Stage", "W-stage (years-based)"),
        ]
        assoc = []
        for var, label in exposure_vars:
            if var not in wd_clean.columns:
                continue
            sub2 = wd_clean.dropna(subset=[var, "PD_Severity_Score"])
            if len(sub2) < 5:
                continue
            rho, pval = stats.spearmanr(sub2[var], sub2["PD_Severity_Score"])
            assoc.append(
                {
                    "variable": var,
                    "label": label,
                    "n": int(len(sub2)),
                    "spearman_rho": float(rho),
                    "p_value": float(pval),
                }
            )
        write_json(metrics_dir / "phase2_associations.json", {"associations": assoc})
    else:
        write_json(
            metrics_dir / "phase2_associations.json",
            {
                "associations": [],
                "note": "Off by default; use --exposure to compute Spearman correlations vs exposure variables.",
            },
        )

    out_xlsx = out_pred / "welder_predictions.xlsx"
    wd_clean.to_excel(out_xlsx, index=False)
    print(f"Phase 2 | {out_xlsx} | figures -> {out_fig}")


def main():
    root = repo_root()
    p = argparse.ArgumentParser(description="Project welders with PD H&Y pipelines.")
    p.add_argument("--data", type=Path, default=default_data_path())
    p.add_argument("--models-dir", type=Path, default=root / "models")
    p.add_argument("--out-fig", type=Path, default=root / "outputs" / "figures")
    p.add_argument("--out-pred", type=Path, default=root / "outputs" / "predictions")
    p.add_argument(
        "--exposure",
        action="store_true",
        help="Secondary: Spearman correlations vs exposure + third panel in fig_07",
    )
    args = p.parse_args()
    if not args.data.is_file():
        print(f"Data not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    if not (args.models_dir / "hy_binary_pipeline.joblib").is_file():
        print("Run: python -m src.train_hy_model", file=sys.stderr)
        sys.exit(1)
    run_projection(
        args.data,
        args.models_dir,
        args.out_fig,
        args.out_pred,
        exposure=args.exposure,
    )


if __name__ == "__main__":
    main()
