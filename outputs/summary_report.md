# Pipeline summary report

Generated from `outputs/metrics/*.json` after `python run_all.py`. Numbers should match `phase1_metrics.json`.

## PD reference models (LOOCV, Combined features)

- **Binary (Early vs Late):** Logistic Regression — accuracy=0.800, balanced_acc=0.778, macro-F1=0.785
- **Multiclass (H&Y I–IV):** Logistic Regression — exact acc=0.400, within-one-stage=0.933, macro-F1=0.222

*Selection rule (Combined):* binary — macro-F1 → balanced accuracy → accuracy; multiclass — macro-F1 → balanced accuracy → within-one-stage accuracy → accuracy (see `docs/model_design.md`).

## Supporting benchmark: PD vs welder (not the H&Y model)

- **Design:** 5-fold_stratified_cv on n=31 (PD=15, welder=16).
- **Best on Combined (by macro-F1):** Random Forest — acc=0.806, bal_acc=0.804, macro-F1=0.805

## Main output paths

| Artifact | Path |
|----------|------|
| Metrics (Phase 1) | `outputs/metrics/phase1_metrics.json` |
| Group benchmark | `outputs/metrics/group_discrimination.json` |
| Welder predictions | `outputs/predictions/welder_predictions.xlsx` |
| Saved pipelines | `models/hy_binary_pipeline.joblib`, `models/hy_multiclass_pipeline.joblib` |
| Schema | `models/schema.json` |

## Main figures (pipeline)

| Figure | Description |
|--------|-------------|
| `outputs/figures/fig_01_pd_balance_by_hy.png` | PD balance scales by H&Y |
| `outputs/figures/fig_02_confusion_binary.png` | LOOCV binary confusion |
| `outputs/figures/fig_03_confusion_multiclass.png` | LOOCV multiclass confusion |
| `outputs/figures/fig_04_cv_summary_binary.png` | LOOCV metrics by feature set |
| `outputs/figures/fig_05_rf_importance_binary.png` | RF feature importance (binary) |
| `outputs/figures/fig_06_pd_vs_welder_balance.png` | PD vs welder balance scales |
| `outputs/figures/fig_07_welder_projection.png` | Welder severity / probabilities (+ exposure panel if `--exposure`) |
| `outputs/figures/fig_08_group_discrimination.png` | Supporting group discrimination |
| `outputs/figures/fig_09_multiclass_confidence_loocv.png` | Multiclass LOOCV confidence (max P / P true stage) |

## Brief interpretation

- **Phase 1** fits **PD-only** models on BBS, Mini-BEST, and FES; LOOCV metrics describe reference-model behavior on *n*≈14 PD patients — hypothesis-generating, not clinical validation.
- **Phase 2** applies the **multiclass** pipeline to welders: outputs are **PD-like resemblance** (stage probabilities and severity score), **not** a PD diagnosis and **not** true H&Y in welders.
- *Exposure correlations:* Off by default; use --exposure to compute Spearman correlations vs exposure variables.
- **Group discrimination** shows separation on shared balance features between cohorts; confounded by age and design — descriptive only.

## Schema snapshot

- Seed: 42 | Features: ['BBS', 'Mini', 'FES']
