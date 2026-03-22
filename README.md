# Motor Balance Analysis: PD Patients vs Welders

> **Pilot study — exploratory analysis only.**
> n = 30 (14 PD, 16 welders). Results are hypothesis-generating, not clinically validated predictions.

---

## Study Overview

This repository contains a two-phase analysis comparing postural balance between:

- **Parkinson's Disease (PD) patients** (n = 14, mean age 71.7 ± 6.2 yrs, Stage I–IV Hoehn & Yahr)
- **Occupational welders** (n = 16, mean age 41.8 ± 6.3 yrs, 8–38 years of exposure)

Three standardised balance scales were administered to both groups:

| Scale | Full Name | Range | Higher = |
|-------|-----------|-------|----------|
| BBS | Berg Balance Scale | 0–56 | Better |
| MiniBESTest | Mini Balance Evaluation Systems Test | 0–28 | Better |
| FES-I | Falls Efficacy Scale – International | 10–40 | Worse (more fear) |

> **Note:** ABC (Activities-specific Balance Confidence) and TUG (Timed Up and Go) scales were **not collected** in this dataset.

---

## Repository Structure

```
motor-balance-pd-welders/
├── Motor_Balance_Analysis_Phase2_Complete.ipynb   ← Main analysis notebook
├── PD_WELDERS RAW Long Data-2.xlsx                ← Raw data (both groups)
├── Motor_Balance_Limitations_ReviewerResponse.docx ← Limitations & reviewer notes
├── Motor_Balance_Analysis copy_executed.ipynb - Colab.pdf  ← Original executed reference
├── .gitignore
└── README.md
```

---

## Notebook Structure

The single notebook (`Motor_Balance_Analysis_Phase2_Complete.ipynb`) contains all analysis in 20 sections:

**Phase 1 (Sections 0–13) — Validity Fixes**

| Section | Content |
|---------|---------|
| 0–2 | Setup, data loading, helper functions |
| 3 | Feature extraction (correct column mapping) |
| 4 | Age confound analysis — Mann-Whitney U, Spearman, visualisation |
| 5–9 | Descriptive stats, W1–W5 staging, correlations, visualisations |
| 10–11 | Feature engineering + LOOCV modelling pipeline |
| 12–13 | Model comparison summary + key findings |

**Phase 2 (Sections 14–20) — Extended Analysis**

| Section | Content |
|---------|---------|
| 14 | Bootstrap 95% CIs on all LOOCV metrics |
| 15 | Age-adjusted classification (residual features per fold) |
| 16 | H&Y binary staging: Early (I–II) vs Late (III–IV) within PD |
| 17 | Welder dose-response: total years × balance scales |
| 18 | Cohen's d effect sizes + Publication Table 1 |
| 19 | ROC-AUC (LOOCV) + Permutation feature importance |
| 20 | Verified results summary |

---

## Key Results (Verified, LOOCV)

### Group Classification: PD vs Welder

Using balance scales only (BBS, MiniBEST, FES — **no Age**):

| Model | Accuracy | 95% CI | F1-macro |
|-------|----------|--------|----------|
| Random Forest | **76.7%** | [60.0–90.0%] | 0.764 |
| Logistic Regression | 73.3% | [56.7–86.7%] | 0.732 |
| Majority baseline | 53.3% | — | — |
| ROC-AUC | **0.839** | — | — |

> Including Age raises accuracy to 100%, but the **30-year age gap** (PD 71.7 vs Welder 41.8 yrs) means Age is the dominant separator, not balance. Age-confounded results are not reported as primary findings.

### Fall Risk Prediction (Independent Label)

Label sourced from clinical records (*Number of Falls in Last 6 Months* for PD; *History of Fall* for welders), **not** derived from scale scores. n = 23 (7 welders had missing fall history).

| Model | Accuracy | 95% CI | F1-macro |
|-------|----------|--------|----------|
| Gradient Boosting | **69.6%** | [52.2–87.0%] | 0.693 |
| Majority baseline | 56.5% | — | — |

> The original notebook reported 96.7% fall-risk accuracy. That used `high_fall_risk = (BBS < 45)` as the label while also feeding BBS as a feature — a circular definition. The 69.6% figure is the corrected, non-circular result.

### H&Y Binary Staging within PD

Early (Stage I–II, n=5) vs Late (Stage III–IV, n=9), LOOCV within 14 PD patients.

| Model | Accuracy | 95% CI | F1-macro |
|-------|----------|--------|----------|
| Logistic Regression | **85.7%** | [64.3–100%] | 0.844 |
| Random Forest | 85.7% | [64.3–100%] | 0.844 |
| Majority baseline | 64.3% | — | — |

> The original 5-class H&Y model collapsed all 16 welders to Stage 2 — a degenerate result from cross-population extrapolation. Binary within-PD classification is the valid approach.

### Welder Dose-Response

Spearman ρ between total welding years and balance scales (within welders, n = 16):

| Scale | ρ | p-value |
|-------|---|---------|
| FES-I | +0.517 | **0.040 \*** |
| BBS | +0.032 | 0.908 ns |
| MiniBESTest | −0.136 | 0.614 ns |

Only FES-I shows a dose-response signal. Longer-exposed welders report significantly greater fear of falling; balance scale scores (BBS, MiniBESTest) do not track exposure duration significantly.

### Effect Sizes (Cohen's d, PD vs Welder)

| Variable | PD | Welder | d | Effect |
|----------|----|--------|---|--------|
| Age | 71.7 ± 6.2 | 41.8 ± 6.3 | +4.80 | Large *** |
| BBS | 34.1 ± 12.7 | 47.4 ± 10.3 | −1.16 | Large ** |
| MiniBESTest | 12.3 ± 4.5 | 19.8 ± 7.1 | −1.24 | Large ** |
| FES-I | 54.1 ± 19.5 | 32.2 ± 25.5 | +0.96 | Large * |

All balance differences are statistically significant and large. However, Age d = 4.80 dwarfs all balance effects — without an age-matched design, balance differences cannot be cleanly attributed to disease vs exposure vs normal ageing.

---

## What This Study Credibly Supports

- Strong descriptive group differences across all three balance scales
- W1–W5 welding exposure staging as an **exploratory conceptual framework**
- FES-I dose-response trend worthy of follow-up in a larger, age-matched sample
- H&Y binary staging within PD is feasible even with small n
- LOOCV is the correct validation approach for n = 30

## Limitations

| Limitation | Impact |
|-----------|--------|
| n = 30 total | Wide confidence intervals throughout; all results should be treated as preliminary |
| 30-year age gap (PD vs welders) | Major confound; partially irremovable without age-matched recruitment |
| 7/16 welders missing fall history | Fall-risk model uses n = 23, not n = 30 |
| Cross-sectional design | No causal inference possible |
| W1–W5 staging schema | Preliminary; not clinically validated |
| Original circular labels | `Motor_Balance_Severity` and `high_fall_risk` were derived from scale scores then used as prediction targets with those same scores as features — removed in Phase 1 |

---

## Running the Notebook

```python
# In Google Colab:
# 1. Upload PD_WELDERS RAW Long Data-2.xlsx when prompted in Section 1
# 2. Runtime → Run all

# Locally:
pip install pandas numpy scikit-learn matplotlib seaborn scipy openpyxl xgboost
jupyter notebook Motor_Balance_Analysis_Phase2_Complete.ipynb
```

---

## Citation / Attribution

Exploratory pilot study. If referencing, please frame results as preliminary and note the sample size and age-confound limitations explicitly.

---

*Analysis conducted with Python 3, scikit-learn, scipy, matplotlib/seaborn.*
*Validation: Leave-One-Out Cross-Validation (LOOCV) throughout.*
*Bootstrap 95% CIs: 1 000 iterations on LOOCV prediction vectors.*
