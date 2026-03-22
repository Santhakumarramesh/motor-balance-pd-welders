# Model design and experiment specification

## Training cohort

- **Source:** PD sheet in the default Excel (`data/PD_WELDERS RAW Long Data.xlsx`; see `src/utils.py`)
- **Sample size:** *n* = 15 PD rows with complete balance data after dropping blank rows (pilot; default data file)
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

Models are compared on the **Combined** feature set (BBS + Mini-BEST + FES). Selection uses the same ordering for all classifier types:

**Binary (Early vs Late):** maximize (macro-F1 → balanced accuracy → accuracy).

**Multiclass (H&Y I–IV):** maximize (macro-F1 → balanced accuracy → within-one-stage accuracy → accuracy).

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

### A. Exposure associations

Welding years, PPE, etc., vs `PD_Severity_Score` (Spearman) — **exploratory**; not the primary claim.

### B. PD vs welder group discrimination (supporting benchmark)

**Not** the H&Y reference model. Binary label: PD (*n* = 15) vs Welder (*n* = 16) on the same three balance features. **5-fold stratified cross-validation**, same pipelines and feature subsets (BBS-only, Mini-only, FES-only, Combined).

Output: `outputs/metrics/group_discrimination.json`, figure `outputs/figures/fig_08_group_discrimination.png`.

**Caveat:** Strong age and cohort confounding — use only to show group separation on balance scores, not causal inference.

## External validation (not part of the pilot results)

Independent PD cohorts with true H&Y are required to test generalization of **frozen** pipelines. See **`docs/external_validation.md`** and `python -m src.validate_external_pd`.

## Reproducibility

- Random seed: **42** (`src/utils.py`)
- Dependencies: pinned in `requirements.txt`
- Run: `python run_all.py` (trains → group benchmark → welder projection → paper figures)
- Metrics: `outputs/metrics/phase1_metrics.json`, `outputs/metrics/group_discrimination.json`
