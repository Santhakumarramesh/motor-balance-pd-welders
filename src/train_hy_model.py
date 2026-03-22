"""PD H&Y reference training: LOOCV, feature ablation, persist fitted Pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib

from src.utils import (
    FEATURES,
    RANDOM_SEED,
    default_data_path,
    load_pd_dataframe,
    repo_root,
    validate_ranges,
    write_json,
)

np.random.seed(RANDOM_SEED)


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot: int = 1000, seed: int = RANDOM_SEED):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    sc = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            sc.append(metric_fn(np.array(y_true)[idx], np.array(y_pred)[idx]))
        except Exception:
            pass
    if not sc:
        return 0.0, 0.0, 0.0
    return float(np.mean(sc)), float(np.percentile(sc, 2.5)), float(np.percentile(sc, 97.5))


def loocv_predict(pipeline_template: Pipeline, X: np.ndarray, y: np.ndarray):
    loo = LeaveOneOut()
    yt, yp = [], []
    for tr, te in loo.split(X):
        est = clone(pipeline_template)
        est.fit(X[tr], y[tr])
        yp.append(est.predict(X[te])[0])
        yt.append(int(y[te][0]))
    return np.array(yt), np.array(yp)


def report_metrics(name: str, yt, yp, multiclass: bool):
    acc = accuracy_score(yt, yp)
    bal = balanced_accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, average="macro", zero_division=0)
    acc1 = float(np.mean(np.abs(yt - yp) <= 1)) if multiclass else None
    _, alo, ahi = bootstrap_ci(yt, yp, accuracy_score)
    _, flo, fhi = bootstrap_ci(
        yt,
        yp,
        lambda a, b: f1_score(a, b, average="macro", zero_division=0),
    )
    return {
        "name": name,
        "accuracy": acc,
        "accuracy_ci_low": alo,
        "accuracy_ci_high": ahi,
        "balanced_accuracy": bal,
        "f1_macro": f1,
        "f1_ci_low": flo,
        "f1_ci_high": fhi,
        "within_one_stage_acc": acc1,
    }


def make_classifier(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=2000,
            C=1.0,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED,
        )
    raise ValueError(name)


def make_pipeline(clf_name: str) -> Pipeline:
    clf = make_classifier(clf_name)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


MODEL_NAMES = ("Logistic Regression", "Random Forest", "Gradient Boosting")

FEATURE_SETS = {
    "BBS": ["BBS"],
    "Mini-BEST": ["Mini"],
    "FES": ["FES"],
    "Combined": list(FEATURES),
}


def select_best_model(metric_rows: list[dict], *, multiclass: bool = False) -> dict:
    if multiclass:

        def mc_key(r: dict):
            w1 = r.get("within_one_stage_acc")
            return (
                r["f1_macro"],
                r["balanced_accuracy"],
                w1 if w1 is not None else 0.0,
                r["accuracy"],
            )

        return max(metric_rows, key=mc_key)
    return max(
        metric_rows,
        key=lambda r: (r["f1_macro"], r["balanced_accuracy"], r["accuracy"]),
    )


def run_phase1(
    data_path: Path,
    out_fig: Path,
    out_metrics: Path,
    models_dir: Path,
):
    pd_df = load_pd_dataframe(data_path)
    for w in validate_ranges(pd_df, "PD"):
        print("Warning:", w)

    out_fig.mkdir(parents=True, exist_ok=True)
    palette = {1: "#3498db", 2: "#2ecc71", 3: "#f39c12", 4: "#e74c3c"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (col, ylabel, worse) in zip(
        axes,
        [
            ("BBS", "BBS Total (/56)", "lower = worse"),
            ("Mini", "Mini-BESTest (/28)", "lower = worse"),
            ("FES", "FES-I Total (/40)", "higher = worse"),
        ],
    ):
        sub = pd_df.dropna(subset=[col, "HY"]).copy()
        sub["HY"] = sub["HY"].astype(int)
        order = sorted(sub["HY"].unique())
        sns.boxplot(
            data=sub,
            x="HY",
            y=col,
            ax=ax,
            hue="HY",
            palette=palette,
            order=order,
            linewidth=1.2,
            fliersize=4,
            dodge=False,
            legend=False,
        )
        sns.stripplot(
            data=sub,
            x="HY",
            y=col,
            ax=ax,
            color="black",
            size=5,
            alpha=0.7,
            order=order,
            jitter=0.1,
        )
        ax.set_xlabel("H&Y Stage")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{col} by H&Y Stage\n({worse})")
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle(
        "Balance Scores by Hoehn & Yahr Stage (PD patients, n=14)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_fig / "fig_01_pd_balance_by_hy.png", dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "binary": {},
        "multiclass": {},
        "feature_sets": list(FEATURE_SETS.keys()),
    }

    df_bin = pd_df.dropna(subset=list(FEATURES) + ["HY_bin"]).reset_index(drop=True)
    y_bin = df_bin["HY_bin"].values.astype(int)
    n_cls = np.bincount(y_bin)
    baseline_bin = float(n_cls.max() / n_cls.sum())

    for fs_name, cols in FEATURE_SETS.items():
        Xb = df_bin[cols].values.astype(float)
        rows = []
        for mname in MODEL_NAMES:
            pipe = make_pipeline(mname)
            yt, yp = loocv_predict(pipe, Xb, y_bin)
            row = report_metrics(mname, yt, yp, multiclass=False)
            rows.append(row)
        results["binary"][fs_name] = {"models": rows, "baseline": baseline_bin}

    df_mc = pd_df.dropna(subset=list(FEATURES) + ["HY"]).reset_index(drop=True)
    y_mc = df_mc["HY"].values.astype(int)
    stages = sorted(np.unique(y_mc))
    labels = [f"Stage {s}" for s in stages]
    n_per = dict(zip(*np.unique(y_mc, return_counts=True)))
    baseline_mc = float(max(n_per.values()) / sum(n_per.values()))

    for fs_name, cols in FEATURE_SETS.items():
        Xm = df_mc[cols].values.astype(float)
        rows = []
        for mname in MODEL_NAMES:
            pipe = make_pipeline(mname)
            yt, yp = loocv_predict(pipe, Xm, y_mc)
            row = report_metrics(mname, yt, yp, multiclass=True)
            rows.append(row)
        results["multiclass"][fs_name] = {"models": rows, "baseline": baseline_mc}

    bin_combined = results["binary"]["Combined"]["models"]
    best_bin_row = select_best_model(bin_combined, multiclass=False)
    best_bin_name = best_bin_row["name"]

    mc_combined = results["multiclass"]["Combined"]["models"]
    best_mc_row = select_best_model(mc_combined, multiclass=True)
    best_mc_name = best_mc_row["name"]

    X_bin_full = df_bin[list(FEATURES)].values.astype(float)
    pipe_bin = make_pipeline(best_bin_name)
    yt_b, yp_b = loocv_predict(pipe_bin, X_bin_full, y_bin)

    X_mc_full = df_mc[list(FEATURES)].values.astype(float)
    pipe_mc = make_pipeline(best_mc_name)
    yt_m, yp_m = loocv_predict(pipe_mc, X_mc_full, y_mc)

    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(yt_b, yp_b)
    ConfusionMatrixDisplay(cm, display_labels=["Early (I-II)", "Late (III-IV)"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Binary H&Y LOOCV — {best_bin_name}")
    plt.tight_layout()
    plt.savefig(out_fig / "fig_02_confusion_binary.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(yt_m, yp_m, labels=stages)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Multiclass H&Y LOOCV — {best_mc_name}")
    plt.tight_layout()
    plt.savefig(out_fig / "fig_03_confusion_multiclass.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    metrics_names = ["accuracy", "balanced_accuracy", "f1_macro"]
    x = np.arange(len(metrics_names))
    width = 0.6
    for ax, fs_name in zip(axes, FEATURE_SETS.keys()):
        mrows = results["binary"][fs_name]["models"]
        best = select_best_model(mrows, multiclass=False)
        vals = [best[m] for m in metrics_names]
        ax.bar(x, vals, width, color="steelblue", edgecolor="k")
        ax.set_xticks(x)
        ax.set_xticklabels(["Accuracy", "Bal. acc.", "F1 macro"])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{fs_name}\n({best['name']})")
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("LOOCV binary H&Y (best model per feature set)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_fig / "fig_04_cv_summary_binary.png", dpi=150, bbox_inches="tight")
    plt.close()

    rf = make_pipeline("Random Forest")
    rf.fit(X_bin_full, y_bin)
    imp = rf.named_steps["clf"].feature_importances_
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(list(FEATURES), imp, color="coral", edgecolor="k")
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("Random Forest — binary model\nfeature importance (full PD fit)")
    plt.tight_layout()
    plt.savefig(out_fig / "fig_05_rf_importance_binary.png", dpi=150, bbox_inches="tight")
    plt.close()

    models_dir.mkdir(parents=True, exist_ok=True)
    final_bin = clone(make_pipeline(best_bin_name))
    final_bin.fit(X_bin_full, y_bin)

    final_mc = clone(make_pipeline(best_mc_name))
    final_mc.fit(X_mc_full, y_mc)

    schema = {
        "random_seed": RANDOM_SEED,
        "features": list(FEATURES),
        "binary": {
            "model_name": best_bin_name,
            "target": "HY_bin",
            "classes": [0, 1],
            "labels": ["Early (I-II)", "Late (III-IV)"],
        },
        "multiclass": {
            "model_name": best_mc_name,
            "target": "HY",
            "classes": [int(s) for s in stages],
            "labels": labels,
        },
        "validation": "LOOCV",
        "preprocessing": ["SimpleImputer(median)", "StandardScaler", "classifier"],
    }
    write_json(models_dir / "schema.json", schema)

    joblib.dump(
        {
            "pipeline": final_bin,
            "features": list(FEATURES),
            "classes": np.array([0, 1]),
            "labels": ["Early (I-II)", "Late (III-IV)"],
            "model_name": best_bin_name,
            "task": "binary_hy",
        },
        models_dir / "hy_binary_pipeline.joblib",
    )
    joblib.dump(
        {
            "pipeline": final_mc,
            "features": list(FEATURES),
            "classes": np.array(stages),
            "labels": labels,
            "model_name": best_mc_name,
            "task": "multiclass_hy",
        },
        models_dir / "hy_multiclass_pipeline.joblib",
    )

    summary = {
        "schema": schema,
        "binary_best_combined": best_bin_row,
        "multiclass_best_combined": best_mc_row,
        "classification_report_binary": classification_report(
            yt_b, yp_b, target_names=["Early (I-II)", "Late (III-IV)"], zero_division=0
        ),
        "classification_report_multiclass": classification_report(
            yt_m, yp_m, target_names=labels, zero_division=0
        ),
        "loocv": results,
    }
    write_json(out_metrics / "phase1_metrics.json", summary)

    print(
        f"Phase 1 | binary: {best_bin_name} | LOOCV acc {best_bin_row['accuracy']:.3f} | "
        f"F1_macro {best_bin_row['f1_macro']:.3f}"
    )
    print(
        f"Phase 1 | multiclass: {best_mc_name} | LOOCV acc {best_mc_row['accuracy']:.3f} | "
        f"within-1-stage {best_mc_row['within_one_stage_acc']:.3f}"
    )
    print(f"Saved: {models_dir}/hy_*_pipeline.joblib | {out_metrics}/phase1_metrics.json")


def main():
    root = repo_root()
    p = argparse.ArgumentParser(description="Train PD H&Y reference models (LOOCV).")
    p.add_argument("--data", type=Path, default=default_data_path(), help="Excel path")
    p.add_argument(
        "--out-fig",
        type=Path,
        default=root / "outputs" / "figures",
        help="Figure output directory",
    )
    p.add_argument(
        "--out-metrics",
        type=Path,
        default=root / "outputs" / "metrics",
    )
    p.add_argument("--models-dir", type=Path, default=root / "models")
    args = p.parse_args()
    if not args.data.is_file():
        print(f"Data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    run_phase1(args.data, args.out_fig, args.out_metrics, args.models_dir)


if __name__ == "__main__":
    main()
