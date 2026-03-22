"""Build paper-ordered figures and matching caption markdown files."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import (
    FEATURES,
    default_data_path,
    load_pd_dataframe,
    load_wd_dataframe,
    repo_root,
)


def save_workflow_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.3, 2.2, 2.2, 1.2, "Excel input\n(BBS, Mini-BEST, FES)"),
        (3.2, 2.2, 2.2, 1.2, "Phase 1\nPD-only training\n(LOOCV)"),
        (6.1, 2.2, 2.2, 1.2, "Saved pipelines\n(binary + multiclass)"),
        (3.2, 0.4, 2.2, 1.2, "Phase 2\nWelder rows\n(same features)"),
        (6.1, 0.4, 2.2, 1.2, "Outputs\nŷ stage, P(stage),\nseverity score"),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, fill=True, facecolor="#ecf0f1", edgecolor="#2c3e50", lw=1.5)
        )
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9, wrap=True)
    ax.annotate("", xy=(3.1, 2.8), xytext=(2.6, 2.8), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(6.0, 2.8), xytext=(5.5, 2.8), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(4.3, 2.15), xytext=(4.3, 1.65), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(6.0, 1.0), xytext=(5.5, 1.0), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title("PD-referenced motor-balance framework (conceptual)", fontweight="bold", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_pd_vs_welder_panels(pd_df: pd.DataFrame, wd_df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for _, r in pd_df.iterrows():
        rows.append({"Group": "PD", "BBS": r["BBS"], "Mini": r["Mini"], "FES": r["FES"]})
    for _, r in wd_df.iterrows():
        rows.append({"Group": "Welder", "BBS": r["BBS"], "Mini": r["Mini"], "FES": r["FES"]})
    long = pd.DataFrame(rows)
    meta = [
        ("BBS", "paper_fig_02_bbs_pd_vs_welder.png", "BBS (0–56; higher = better)"),
        ("Mini", "paper_fig_03_minibest_pd_vs_welder.png", "Mini-BEST (0–28; higher = better)"),
        ("FES", "paper_fig_04_fes_pd_vs_welder.png", "FES (higher = worse fear of falling)"),
    ]
    for col, fname, title in meta:
        fig, ax = plt.subplots(figsize=(4.5, 4))
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
        sns.stripplot(data=sub, x="Group", y=col, ax=ax, color="k", alpha=0.5, size=5, jitter=0.12)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()


def welder_supplementary_figures(pred: pd.DataFrame, out_dir: Path) -> None:
    if "Pred_Stage" not in pred.columns:
        return
    vc = pred["Pred_Stage"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    vc.plot(kind="bar", ax=ax, color="steelblue", edgecolor="k")
    ax.set_xlabel("Predicted PD-like stage")
    ax.set_ylabel("Count (welders)")
    ax.set_title("Distribution of predicted PD-like H&Y stage")
    plt.tight_layout()
    plt.savefig(out_dir / "paper_fig_08_welder_stage_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    mc_cols = [c for c in pred.columns if c.startswith("P_Stage")]
    if len(mc_cols) < 2:
        return
    stages = sorted(int(c.replace("P_Stage", "")) for c in mc_cols)
    colors = {1: "#3498db", 2: "#2ecc71", 3: "#f39c12", 4: "#e74c3c"}
    fig, ax = plt.subplots(figsize=(10, 4))
    bottoms = np.zeros(len(pred))
    x = np.arange(len(pred))
    for s in stages:
        col = f"P_Stage{s}"
        if col in pred.columns:
            ax.bar(x, pred[col], bottom=bottoms, label=f"Stage {s}", color=colors.get(s, "#888"), alpha=0.9)
            bottoms += pred[col].values
    ax.set_xticks(x)
    ax.set_xticklabels(pred["ID"].astype(str).values, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Probability")
    ax.set_title("Per-welder PD-like stage probabilities (multiclass model)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "paper_fig_09_welder_stage_probabilities.png", dpi=150, bbox_inches="tight")
    plt.close()

    if "PD_Severity_Score" in pred.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(pred["PD_Severity_Score"], bins=10, color="steelblue", edgecolor="k", alpha=0.85)
        ax.axvline(pred["PD_Severity_Score"].mean(), color="r", linestyle="--", label="Mean")
        ax.set_xlabel("PD-like severity score (1–4)")
        ax.set_ylabel("Welders")
        ax.set_title("PD-like severity score distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "paper_fig_10_welder_severity_score.png", dpi=150, bbox_inches="tight")
        plt.close()


def load_metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_caption_md(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")


def ensure_pipeline_outputs(root: Path) -> None:
    m = root / "outputs" / "metrics" / "phase1_metrics.json"
    if m.is_file():
        return
    subprocess.check_call([sys.executable, str(root / "run_all.py")], cwd=root)


def main() -> None:
    root = repo_root()
    p = argparse.ArgumentParser(description="Paper figure set + markdown captions.")
    p.add_argument("--data", type=Path, default=default_data_path())
    p.add_argument(
        "--out",
        type=Path,
        default=root / "outputs" / "figures" / "paper",
        help="Directory for paper_fig_*.png and .md",
    )
    p.add_argument(
        "--ensure-run",
        action="store_true",
        help="Run run_all.py if metrics/figures are missing",
    )
    args = p.parse_args()

    if args.ensure_run:
        ensure_pipeline_outputs(root)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_src = root / "outputs" / "figures"
    metrics_path = root / "outputs" / "metrics" / "phase1_metrics.json"
    met = load_metrics(metrics_path)

    save_workflow_diagram(out_dir / "paper_fig_01_workflow.png")
    write_caption_md(
        out_dir / "paper_fig_01_workflow.md",
        "Figure 1 — Study workflow",
        "Conceptual flow: shared balance features from Excel → PD-only model fitting with LOOCV → "
        "saved estimators -> application to welder rows to obtain PD-like stage probabilities and "
        "severity score. This is a resemblance framework, not a PD diagnosis in welders.",
    )

    if args.data.is_file():
        pd_df = load_pd_dataframe(args.data)
        wd_df = load_wd_dataframe(args.data)
        save_pd_vs_welder_panels(pd_df, wd_df, out_dir)
        for fname, title, body in [
            (
                "paper_fig_02_bbs_pd_vs_welder.md",
                "Figure 2 — BBS (PD vs welders)",
                "Boxplots of Berg Balance Scale scores by group. Higher scores indicate better balance. "
                "Descriptive comparison only; groups differ in age and disease context.",
            ),
            (
                "paper_fig_03_minibest_pd_vs_welder.md",
                "Figure 3 — Mini-BESTest (PD vs welders)",
                "Mini-BESTest totals by group. Captures dynamic balance domains.",
            ),
            (
                "paper_fig_04_fes_pd_vs_welder.md",
                "Figure 4 — FES (PD vs welders)",
                "Falls Efficacy Scale; higher scores indicate greater fear of falling / lower confidence.",
            ),
        ]:
            write_caption_md(out_dir / fname, title, body)

    pairs = [
        ("fig_04_cv_summary_binary.png", "paper_fig_05_cv_feature_sets.png"),
        ("fig_02_confusion_binary.png", "paper_fig_06_confusion_binary.png"),
        ("fig_03_confusion_multiclass.png", "paper_fig_07_confusion_multiclass.png"),
        ("fig_05_rf_importance_binary.png", "paper_fig_11_rf_feature_importance.png"),
    ]
    for src_name, dst_name in pairs:
        src = fig_src / src_name
        if src.is_file():
            shutil.copy2(src, out_dir / dst_name)

    bb = met.get("binary_best_combined", {})
    mc = met.get("multiclass_best_combined", {})
    write_caption_md(
        out_dir / "paper_fig_05_cv_feature_sets.md",
        "Figure 5 — LOOCV binary performance by feature set",
        f"Best model per feature subset (BBS-only, Mini-BEST-only, FES-only, Combined). "
        f"Bars: accuracy, balanced accuracy, macro-F1. "
        f"Combined-model selection for deployment: best macro-F1 among classifiers on Combined features "
        f"(binary best: {bb.get('name', '—')}).",
    )
    write_caption_md(
        out_dir / "paper_fig_06_confusion_binary.md",
        "Figure 6 — Binary LOOCV confusion matrix",
        "Early (I–II) vs Late (III–IV). "
        f"LOOCV accuracy ≈ {bb.get('accuracy', 0):.3f}; macro-F1 ≈ {bb.get('f1_macro', 0):.3f}.",
    )
    write_caption_md(
        out_dir / "paper_fig_07_confusion_multiclass.md",
        "Figure 7 — Multiclass LOOCV confusion matrix",
        f"Stages I–IV. Exact accuracy ≈ {mc.get('accuracy', 0):.3f}; "
        f"within-one-stage accuracy ≈ {mc.get('within_one_stage_acc', 0):.3f}.",
    )
    write_caption_md(
        out_dir / "paper_fig_11_rf_feature_importance.md",
        "Figure 11 — Random forest feature importance (binary)",
        "Gini importance from a random forest fit on the full PD set for the binary task; "
        "illustrates relative contribution of BBS, Mini-BEST, and FES in this pilot sample.",
    )

    pred_path = root / "outputs" / "predictions" / "welder_predictions.xlsx"
    if pred_path.is_file():
        pred = pd.read_excel(pred_path)
        welder_supplementary_figures(pred, out_dir)
        write_caption_md(
            out_dir / "paper_fig_08_welder_stage_distribution.md",
            "Figure 8 — Predicted PD-like stage (welders)",
            "Counts of argmax stage from the multiclass PD reference model. Exploratory, non-diagnostic.",
        )
        write_caption_md(
            out_dir / "paper_fig_09_welder_stage_probabilities.md",
            "Figure 9 — Stage probabilities per welder",
            "Stacked predicted probabilities across H&Y stages; conveys uncertainty and mixed resemblance.",
        )
        write_caption_md(
            out_dir / "paper_fig_10_welder_severity_score.md",
            "Figure 10 — PD-like severity score",
            "Distribution of ∑ k·P(stage k); continuous summary for correlation-style secondary analyses.",
        )

    print(f"Paper figures and captions -> {out_dir}")


if __name__ == "__main__":
    main()
