# Figure plan (paper)

Figures produced by the Python pipeline live under `outputs/figures/` after `python run_all.py`.

| File | Content |
|------|---------|
| `fig_01_pd_balance_by_hy.png` | PD balance scales by H&Y stage |
| `fig_02_confusion_binary.png` | LOOCV confusion — binary Early vs Late |
| `fig_03_confusion_multiclass.png` | LOOCV confusion — stages I–IV |
| `fig_04_cv_summary_binary.png` | LOOCV metrics by feature set (BBS, Mini-BEST, FES, Combined) |
| `fig_05_rf_importance_binary.png` | Random Forest feature importance (binary, full PD fit) |
| `fig_06_pd_vs_welder_balance.png` | PD vs welder boxplots (BBS, Mini-BEST, FES) |
| `fig_07_welder_projection.png` | Welder PD-like severity score + probabilities + exposure scatter |

Welder-level predictions: `outputs/predictions/welder_predictions.xlsx`.  
Metrics: `outputs/metrics/phase1_metrics.json`, `outputs/metrics/phase2_associations.json`.
