# Motor Balance Analysis: PD Severity Reference Model + Welder Projection

> **Pilot study — exploratory analysis only.**
> n = 30 (14 PD, 16 welders). Results are hypothesis-generating, not clinically validated predictions.

---

## Research Question

> Can welding-exposed workers be positioned relative to Parkinson's Disease H&Y motor-balance severity patterns — and does resemblance to more severe PD profiles correlate with heavier welding exposure?

This is a **PD severity reference → welder projection** design, not a PD vs welder classification.

---

## Study Design

```
Step 1 (Phase 1):  Train H&Y classifier on PD patients only
                   BBS + MiniBESTest + FES → Stage I / II / III / IV

Step 2 (Phase 2):  Apply PD model to welders
                   Each welder gets: predicted PD-like severity class
                                     stage probabilities
                                     PD-like severity score (1–4)

Step 3 (Phase 2):  Test associations
                   Predicted severity score vs welding years, hours/day,
                   fume/vibration/noise exposure, PPE use, fall history
```

**Why no healthy controls?** The core question is whether welders resemble PD severity patterns — that is a PD vs welder question, not a PD vs welder vs controls question. Including controls would answer a different (less novel) question.

---

## Data

**File:** `PD_WELDERS RAW Long Data-2.xlsx`

| Group | n | Mean Age | H&Y / Exposure |
|-------|---|----------|----------------|
| PD patients | 14 | 71.7 ± 6.2 yrs | Stage I=2, II=3, III=6, IV=3 |
| Welders | 16 | 41.8 ± 6.3 yrs | 8–38 years welding |

**Balance scales collected (both groups):**

| Scale | Range | Higher = |
|-------|-------|----------|
| BBS (Berg Balance Scale) | 0–56 | Better |
| Mini-BESTest | 0–28 | Better |
| FES-I (Falls Efficacy Scale) | 10–40 | Worse |

> ABC and TUG scales were **not collected** in this dataset.

**Key limitation:** 30-year age gap between groups (PD 71.7 vs Welder 41.8 yrs). Age is a major confound that cannot be fully removed in a cross-sectional design.

---

## Repository Structure

```
motor-balance-pd-welders/
├── phase1_pd_hy_model.ipynb         ← Train PD H&Y severity classifier
├── phase2_welder_projection.ipynb   ← Project welders + exposure associations
├── PD_WELDERS RAW Long Data-2.xlsx  ← Raw data
├── .gitignore
└── README.md
```

### Run Order

```
1. phase1_pd_hy_model.ipynb
   → outputs: pd_hy_binary_model.pkl
              pd_hy_multiclass_model.pkl
              figure1_pd_balance_by_hy.png
              figure2_confusion_binary.png
              figure3_confusion_multiclass.png

2. phase2_welder_projection.ipynb   (requires the .pkl files from Phase 1)
   → outputs: figure4_welder_projection.png
```

---

## Phase 1 Summary — PD H&Y Severity Reference Model

**Task:** Classify PD patients into H&Y stage using balance scores (BBS, MiniBEST, FES).
**Validation:** Leave-One-Out Cross-Validation (LOOCV) + bootstrap 95% CIs.

### Model A — Binary: Early (Stage I–II) vs Late (Stage III–IV)

| Model | Accuracy | 95% CI | F1-macro |
|-------|----------|--------|----------|
| Logistic Regression | **85.7%** | [64.3–100%] | 0.844 |
| Random Forest | 85.7% | [64.3–100%] | 0.844 |
| Majority baseline | 64.3% | — | — |

### Model A2 — Multi-Class (Stages I–IV)

With only 2–6 patients per stage, exact accuracy is limited. **Within-1-stage accuracy** is the more meaningful metric (whether predictions land in the adjacent stage).

> ⚠ Wide confidence intervals throughout — n=14 is too small for stable multi-class estimates.

---

## Phase 2 Summary — Welder Projection + Associations

**Model B:** Each welder receives a PD-like severity score (1–4 weighted probability) indicating which PD severity profile their balance pattern most resembles.

**Model C:** Spearman correlations between predicted severity score and:
- Total years in welding
- Hours per day
- Fume / vibration / noise exposure intensity
- Respiratory PPE use
- History of falls
- W-Stage (exploratory exposure stage)

### W1–W5 Exposure Staging *(Exploratory)*

| Stage | Years | Label |
|-------|-------|-------|
| W1 | 0–<10 | Early exposure |
| W2 | 10–<20 | Moderate exposure |
| W3 | 20–<30 | Significant exposure |
| W4 | 30–<40 | Heavy exposure |
| W5 | ≥40 | Very heavy exposure |

> W1–W5 is a **preliminary conceptual framework**, not a clinically validated staging system.

---

## What This Study Credibly Supports

- PD severity reference model is feasible with small n using binary H&Y classification
- Welder projection into PD severity space as a methodological framework for future work
- FES-I shows a dose-response trend with welding years (ρ=+0.517, p=0.040)
- BBS and MiniBESTest do not show significant dose-response with welding years
- Strong descriptive group differences across all balance scales (all large effect sizes)

## Limitations

| Limitation | Impact |
|-----------|--------|
| n=14 PD, n=16 welders | Very wide CIs; all findings are preliminary |
| 30-year age gap | Major confound; balance differences cannot be attributed solely to disease vs exposure |
| 7/16 welders missing fall history | Fall-history associations use n=9 |
| Only 3 balance scales (no ABC, no TUG) | Incomplete balance profile |
| Cross-sectional design | No causal inference; no longitudinal trajectory |
| W1–W5 schema | Not validated; based on years alone, ignores intensity |
| PD reference model trained on n=14 | Projection to welders is exploratory |

---

## Running the Notebooks

```python
# Google Colab:
# Upload PD_WELDERS RAW Long Data-2.xlsx when prompted in each notebook
# Runtime → Run all (Phase 1 first, then Phase 2)

# Local:
pip install pandas numpy scikit-learn matplotlib seaborn scipy openpyxl joblib
jupyter notebook phase1_pd_hy_model.ipynb
# then:
jupyter notebook phase2_welder_projection.ipynb
```

---

## Honest Framing

This study develops a **PD-referenced motor-balance framework** by training a model on Parkinson's H&Y severity and projecting welders into that learned severity space, to examine whether welding exposure is associated with PD-like motor impairment patterns.

Results suggest strong balance differences between groups and a feasible projection methodology, but the 30-year age gap, small n, and cross-sectional design mean that findings should be treated as **hypothesis-generating pilot results** requiring replication with a larger, age-matched cohort.

---

*Validation: Leave-One-Out Cross-Validation (LOOCV) throughout.*
*Bootstrap 95% CIs: 1 000 iterations on LOOCV prediction vectors.*
*Non-parametric tests (Mann-Whitney U, Spearman) used throughout given small n.*
