"""Write human-readable outputs/summary_report.md from pipeline JSON artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import repo_root


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _best_group_combined(gd: dict) -> tuple[str, str, float, float]:
    """Return (classifier_name, metric_line) for best macro-F1 on Combined."""
    fs = gd.get("feature_sets", {}).get("Combined", {})
    if not fs:
        return "—", "—", 0.0, 0.0
    best_name, best = max(fs.items(), key=lambda kv: kv[1].get("f1_macro", 0.0))
    m = best
    return (
        best_name,
        f"acc={m.get('accuracy', 0):.3f}, bal_acc={m.get('balanced_accuracy', 0):.3f}, "
        f"macro-F1={m.get('f1_macro', 0):.3f}",
        float(m.get("accuracy", 0)),
        float(m.get("f1_macro", 0)),
    )


def write_report(root: Path) -> Path:
    met = _load(root / "outputs" / "metrics" / "phase1_metrics.json")
    gd = _load(root / "outputs" / "metrics" / "group_discrimination.json")
    p2 = _load(root / "outputs" / "metrics" / "phase2_associations.json")

    bb = met.get("binary_best_combined", {})
    mc = met.get("multiclass_best_combined", {})
    sch = met.get("schema", {})

    g_name, g_line, _, _ = _best_group_combined(gd)

    assoc = p2.get("associations") or []
    if assoc:
        exposure_block = (
            f"*Exposure correlations (exploratory):* {len(assoc)} variable(s) tested — "
            f"see `outputs/metrics/phase2_associations.json`."
        )
    elif p2.get("note"):
        exposure_block = f"*Exposure correlations:* {p2['note']}"
    else:
        exposure_block = (
            "*Exposure correlations:* not computed; use `python -m src.project_welders --exposure`."
        )

    lines = [
        "# Pipeline summary report",
        "",
        "Generated from `outputs/metrics/*.json` after `python run_all.py`. "
        "Numbers should match `phase1_metrics.json`.",
        "",
        "## PD reference models (LOOCV, Combined features)",
        "",
        f"- **Binary (Early vs Late):** {bb.get('name', '—')} — "
        f"accuracy={bb.get('accuracy', 0):.3f}, balanced_acc={bb.get('balanced_accuracy', 0):.3f}, "
        f"macro-F1={bb.get('f1_macro', 0):.3f}",
        f"- **Multiclass (H&Y I–IV):** {mc.get('name', '—')} — "
        f"exact acc={mc.get('accuracy', 0):.3f}, within-one-stage={mc.get('within_one_stage_acc', 0):.3f}, "
        f"macro-F1={mc.get('f1_macro', 0):.3f}",
        "",
        "*Selection rule (Combined):* binary — macro-F1 → balanced accuracy → accuracy; "
        "multiclass — macro-F1 → balanced accuracy → within-one-stage accuracy → accuracy "
        "(see `docs/model_design.md`).",
        "",
        "## Supporting benchmark: PD vs welder (not the H&Y model)",
        "",
        f"- **Design:** {gd.get('validation', '5-fold CV')} on n={gd.get('n_total', '—')} "
        f"(PD={gd.get('n_pd')}, welder={gd.get('n_welder')}).",
        f"- **Best on Combined (by macro-F1):** {g_name} — {g_line}",
        "",
        "## Main output paths",
        "",
        "| Artifact | Path |",
        "|----------|------|",
        "| Metrics (Phase 1) | `outputs/metrics/phase1_metrics.json` |",
        "| Group benchmark | `outputs/metrics/group_discrimination.json` |",
        "| Welder predictions | `outputs/predictions/welder_predictions.xlsx` |",
        "| Saved pipelines | `models/hy_binary_pipeline.joblib`, `models/hy_multiclass_pipeline.joblib` |",
        "| Schema | `models/schema.json` |",
        "",
        "## Main figures (pipeline)",
        "",
        "| Figure | Description |",
        "|--------|-------------|",
        "| `outputs/figures/fig_01_pd_balance_by_hy.png` | PD balance scales by H&Y |",
        "| `outputs/figures/fig_02_confusion_binary.png` | LOOCV binary confusion |",
        "| `outputs/figures/fig_03_confusion_multiclass.png` | LOOCV multiclass confusion |",
        "| `outputs/figures/fig_04_cv_summary_binary.png` | LOOCV metrics by feature set |",
        "| `outputs/figures/fig_05_rf_importance_binary.png` | RF feature importance (binary) |",
        "| `outputs/figures/fig_06_pd_vs_welder_balance.png` | PD vs welder balance scales |",
        "| `outputs/figures/fig_07_welder_projection.png` | Welder severity / probabilities (+ exposure panel if `--exposure`) |",
        "| `outputs/figures/fig_08_group_discrimination.png` | Supporting group discrimination |",
        "| `outputs/figures/fig_09_multiclass_confidence_loocv.png` | Multiclass LOOCV confidence (max P / P true stage) |",
        "",
        "## Brief interpretation",
        "",
        "- **Phase 1** fits **PD-only** models on BBS, Mini-BEST, and FES; LOOCV metrics describe "
        "reference-model behavior on *n*≈14 PD patients — hypothesis-generating, not clinical validation.",
        "- **Phase 2** applies the **multiclass** pipeline to welders: outputs are **PD-like resemblance** "
        "(stage probabilities and severity score), **not** a PD diagnosis and **not** true H&Y in welders.",
        f"- {exposure_block}",
        "- **Group discrimination** shows separation on shared balance features between cohorts; "
        "confounded by age and design — descriptive only.",
        "",
    ]

    if sch:
        lines.extend(
            [
                "## Schema snapshot",
                "",
                f"- Seed: {sch.get('random_seed', '—')} | Features: {sch.get('features', [])}",
                "",
            ]
        )

    out = root / "outputs" / "summary_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Write outputs/summary_report.md from metrics JSON.")
    p.add_argument("--root", type=Path, default=None, help="Repo root (default: auto)")
    args = p.parse_args()
    root = args.root or repo_root()
    path = write_report(root)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
