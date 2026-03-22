# Motor Balance Analysis: PD-Referenced Severity Model + Welder Projection

> **Pilot study — exploratory analysis only.**  
> n = 30 (14 PD, 16 welders). Results are hypothesis-generating, not clinically validated predictions.

**Repository:** [github.com/Santhakumarramesh/motor-balance-pd-welders](https://github.com/Santhakumarramesh/motor-balance-pd-welders)

---

## Research question

> Can welding-exposed workers be positioned relative to Parkinson’s Disease H&Y motor-balance severity patterns — and does resemblance to more severe PD profiles correlate with heavier welding exposure?

This is a **PD severity reference → welder projection** design. Welders receive a **PD-like** motor-balance resemblance profile, not a clinical PD diagnosis or true H&Y staging.

---

## Quick start (recommended)

From the **repository root** (directory that contains `run_all.py`):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py
```

This runs, in order:

1. `python -m src.train_hy_model` — PD-only LOOCV, feature-set ablation, saves **sklearn Pipelines** under `models/`, metrics under `outputs/metrics/`, figures under `outputs/figures/`.
2. `python -m src.project_welders` — projects welders, writes `outputs/predictions/welder_predictions.xlsx` and further figures.

**Data file (required):** `data/PD_WELDERS RAW Long Data-2.xlsx`

**Random seed:** `42` (see `src/utils.py` and `models/schema.json` after training).

---

## Repository layout

```
motor-balance-pd-welders/
├── data/
│   └── PD_WELDERS RAW Long Data-2.xlsx
├── src/
│   ├── utils.py              # Loading, parsing, column normalization
│   ├── train_hy_model.py     # Phase 1 — LOOCV + save Pipelines
│   └── project_welders.py      # Phase 2 — welder projection + associations
├── models/                   # Generated: *.joblib, schema.json (not all tracked in git)
├── outputs/
│   ├── figures/              # fig_01 … fig_07
│   ├── metrics/              # phase1_metrics.json, phase2_associations.json
│   └── predictions/          # welder_predictions.xlsx
├── notebooks/                # Optional interactive workflow (Colab-friendly)
├── docs/paper_figure_plan.md
├── run_all.py
├── requirements.txt
└── README.md
```

---

## Model pipeline (scripts)

- **Inputs (PD training):** `BBS`, `Mini` (Mini-BEST), `FES` — same columns for welders at inference.
- **Preprocessing (inside each CV fold / full fit):** `SimpleImputer(median)` → `StandardScaler` → classifier.
- **Targets:**  
  - Binary: Early H&Y I–II vs Late III–IV.  
  - Multiclass: stages I–IV (used for welder class probabilities and PD-like severity score \(\sum_k P(\text{stage } k)\times k\)).
- **Validation:** Leave-one-out CV (LOOCV) on PD only; bootstrap 95% CIs for accuracy / macro-F1 on LOOCV vectors.
- **Algorithms compared:** Logistic Regression, Random Forest, Gradient Boosting. **Production** artifacts use the **best macro-F1** model on the **Combined** feature set (see `outputs/metrics/phase1_metrics.json`).

---

## Notebooks (optional)

- `notebooks/phase1_pd_hy_model.ipynb` — mirrors Phase 1; saves **legacy** `pd_hy_*.pkl` (scaler + clf) under `../models/` when run with kernel cwd = `notebooks/`.
- `notebooks/phase2_welder_projection.ipynb` — loads **`../models/hy_*_pipeline.joblib`** if present (from `run_all.py`), else legacy `../models/pd_hy_*.pkl`.

For Colab, upload the Excel file when prompted; on local Jupyter, paths assume notebooks live in `notebooks/`.

---

## Phase 1 summary (illustrative; see `phase1_metrics.json` after you run)

| Task | Notes |
|------|--------|
| Binary Early vs Late | LOOCV; compare with majority baseline |
| Multiclass I–IV | Prefer **within-one-stage** accuracy when interpreting |
| Feature ablation | BBS-only, Mini-BEST-only, FES-only, Combined (see `fig_04_cv_summary_binary.png`) |

---

## Phase 2 outputs

- Per-welder: predicted binary class, multiclass stage, stage probabilities, **PD-like severity score** (1–4).
- Spearman associations vs exposure variables → `outputs/metrics/phase2_associations.json`.

---

## Limitations

| Issue | Impact |
|-------|--------|
| n = 14 PD, 16 welders | Wide CIs; exploratory only |
| Large age gap (PD vs welders) | Major confound |
| Three balance scales only | No ABC / TUG |
| PD-referenced welder labels | Resemblance, not diagnosis |

---

## Development

```bash
python -m src.train_hy_model --help
python -m src.project_welders --help
```

---

## Honest framing

This study develops a **PD-referenced motor-balance framework** by training on Parkinson’s H&Y severity and projecting welders into that learned space. Findings are **hypothesis-generating** and require replication in a larger, age-matched cohort.

---

*Validation: LOOCV on PD. Bootstrap: 1000 resamples on LOOCV prediction vectors. Non-parametric tests (Spearman) used for associations where appropriate.*
