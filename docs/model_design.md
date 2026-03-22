# Model design and experiment specification

## Training cohort

- **Source:** PD sheet in `data/PD_WELDERS RAW Long Data-2.xlsx`
- **Sample size:** *n* = 14 (pilot)
- **Label (multiclass):** Hoehn & Yahr stage I–IV (parsed from text fields)
- **Label (binary):** Early = stages I–II; Late = stages III–IV

## Input features (shared with welder inference)

| Column | Concept |
|--------|---------|
| BBS | Berg Balance Scale |
| Mini | Mini-BESTest (column auto-detected) |
| FES | Falls Efficacy Scale |

Preprocessing within each LOOCV fold and in the final full-fit: **median imputation → standard scaling → classifier**.

## Algorithms

- Logistic regression (linear baseline)
- Random forest (nonlinear ensemble)
- Gradient boosting (boosted trees)

Compared on the **same** pipeline structure.

## Validation

- **Primary:** Leave-one-out cross-validation (LOOCV) on PD only
- **Uncertainty:** Bootstrap 95% intervals on LOOCV prediction vectors (accuracy, macro-F1)

## Feature-set comparisons

For each of binary and multiclass tasks, models are fit on:

- BBS only  
- Mini-BEST only  
- FES only  
- Combined (BBS + Mini-BEST + FES)

## Production model selection (saved artifacts)

- **Criterion:** Maximum **macro-F1** on the **Combined** feature set, among the three algorithms.
- **Tie-break:** Higher balanced accuracy, then higher exact accuracy (implicit in `select_best_model` ordering).

Saved files:

- `models/hy_binary_pipeline.joblib`
- `models/hy_multiclass_pipeline.joblib`

The **multiclass** pipeline is the primary vehicle for welder **stage probabilities** and **PD-like severity score** (probability-weighted stage index).

## Welder inference (Phase 2)

- **Input:** WD sheet (or any sheet with resolvable BBS / Mini-BEST / FES columns), same three features.
- **Output (per row):**
  - `Pred_Stage` (argmax of multiclass probabilities)
  - `P_Stage1` … `P_Stage4` (as applicable to trained classes)
  - `PD_Severity_Score` = Σ *k* · P(stage *k*)
  - Optional binary columns if `hy_binary_pipeline` is applied

**Interpretation:** Outputs describe **PD-like motor-balance resemblance** in the space learned from PD patients. They **do not** assign a clinical PD diagnosis or a true H&Y stage in welders.

## Secondary analyses (optional)

Exposure variables (e.g. welding years, PPE) and Spearman correlations with `PD_Severity_Score` are **exploratory** and should not be presented as the primary study claim.

## Reproducibility

- Random seed: **42** (`src/utils.py`)
- Run: `python run_all.py` then `python -m src.generate_paper_figures`
- Metrics: `outputs/metrics/phase1_metrics.json`
