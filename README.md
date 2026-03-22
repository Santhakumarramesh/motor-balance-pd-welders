# Motor Balance Analysis: PD-Referenced Severity Model + Welder Projection

> **Pilot study — exploratory analysis only.**  
> n = 30 (14 PD, 16 welders). Hypothesis-generating, not clinically validated.

**Repository:** [github.com/Santhakumarramesh/motor-balance-pd-welders](https://github.com/Santhakumarramesh/motor-balance-pd-welders)

**Independent review framing (exploratory vs. clinical):** **`docs/reviewer_brief.md`**

---

## Primary research design

**Main claim:** Train a **PD-only** H&Y reference model on **BBS, Mini-BEST, and FES** (leave-one-out CV), save the best pipelines, then **classify welder rows into PD-like motor-balance severity categories** using multiclass stage probabilities and a **PD-like severity score**. This is **resemblance in a PD-learned space**, not a diagnosis of PD in welders and not a true clinical H&Y stage in welders.

**Secondary (optional):** Correlations between PD-like severity and welding exposure variables are **exploratory**. They are **off by default**; enable with `python run_all.py --exposure` or `python -m src.project_welders --exposure`. See `outputs/metrics/phase2_associations.json`. Do not treat them as the headline finding.

---

## Quick start

From the **repository root**:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs:

1. `python -m src.train_hy_model` — PD LOOCV, feature ablation, saves `models/hy_*_pipeline.joblib`, `outputs/metrics/phase1_metrics.json`, `outputs/figures/fig_01`–`fig_05` and `fig_09_multiclass_confidence_loocv.png`.
2. `python -m src.benchmark_group_discrimination` — **supporting** PD vs welder group benchmark (5-fold CV), `outputs/metrics/group_discrimination.json`, `fig_08_group_discrimination.png`.
3. `python -m src.project_welders` — welder projection, `outputs/predictions/welder_predictions.xlsx`, `fig_06`–`fig_07`.
4. `python -m src.generate_paper_figures` — paper-ordered PNGs in `outputs/figures/paper/`; caption `.md` files in `docs/captions/`.
5. `python -m src.write_summary_report` — writes `outputs/summary_report.md` (also runs automatically at the end of `run_all.py`).

Use `python run_all.py --exposure` if you want Phase 2 exposure correlations and the third panel of `fig_07_welder_projection.png`.

**Data:** `data/PD_WELDERS RAW Long Data-2.xlsx`  
**Seed:** `42` (`src/utils.py`)

---

## Other entry points

| Command | Purpose |
|---------|---------|
| `python -m src.predict_excel INPUT.xlsx` | Inference on **any** Excel with BBS / Mini-BEST / FES columns (auto-detected); writes `outputs/predictions/inference_predictions.xlsx` by default |
| `python -m src.benchmark_group_discrimination` | Supporting PD vs welder 5-fold CV benchmark |
| `python -m src.generate_paper_figures --ensure-run` | Build paper PNGs + `docs/captions/*.md`; optional full pipeline |
| `python -m src.write_summary_report` | Regenerate `outputs/summary_report.md` from metrics JSON |
| `python -m src.smoke_test` | Quick checks: data load, train, inference |
| `python -m src.validate_external_pd FILE.xlsx` | **External PD cohort:** frozen pipelines, metrics JSON (requires true H&Y) |

Design reference: **`docs/model_design.md`**. Model card: **`docs/model_card.md`**. Reviewer framing: **`docs/reviewer_brief.md`**.  
External validation protocol and paper wording: **`docs/external_validation.md`**.  
Narrative summary: **`results.txt`**; one-page run overview: **`outputs/summary_report.md`** (align numbers with `phase1_metrics.json`).

**When you have an independent PD Excel** (same column layout as the PD sheet, true H&Y):

```bash
python -m src.validate_external_pd /path/to/external_pd.xlsx --sheet PD
```

This applies **frozen** `models/hy_*_pipeline.joblib` and writes `outputs/metrics/external_validation.json`. It does **not** replace prospective external data collection.

---

## Repository layout

```
motor-balance-pd-welders/
├── data/
│   └── PD_WELDERS RAW Long Data-2.xlsx
├── src/
│   ├── utils.py
│   ├── train_hy_model.py
│   ├── benchmark_group_discrimination.py
│   ├── project_welders.py
│   ├── predict_excel.py
│   ├── validate_external_pd.py
│   ├── generate_paper_figures.py
│   ├── write_summary_report.py
│   └── smoke_test.py
├── models/                   # Generated *.joblib (gitignored)
├── outputs/
│   ├── figures/              # Pipeline fig_01–fig_09 (incl. fig_08 benchmark)
│   ├── figures/paper/        # paper_fig_*.png (captions: docs/captions/)
│   ├── metrics/              # phase1_metrics.json, phase2_associations.json
│   ├── predictions/
│   └── summary_report.md     # Generated human-readable run summary
├── notebooks/
├── docs/
│   ├── captions/             # paper_fig_*.md (figure captions)
│   ├── model_design.md
│   ├── model_card.md
│   ├── reviewer_brief.md
│   ├── external_validation.md
│   └── paper_figure_plan.md
├── results.txt
├── run_all.py
├── requirements.txt
└── README.md
```

---

## Model pipeline (summary)

- **Features:** `BBS`, `Mini`, `FES` — same for PD training and welder inference.
- **Preprocessing:** `SimpleImputer(median)` → `StandardScaler` → classifier (inside each LOOCV fold).
- **Production selection (Combined features only):** Among logistic regression, random forest, and gradient boosting, pick the best model using the **same ordering** as in **`docs/model_design.md`**:
  - **Binary (Early vs Late):** maximize **macro-F1 → balanced accuracy → accuracy**.
  - **Multiclass (H&Y I–IV):** maximize **macro-F1 → balanced accuracy → within-one-stage accuracy → accuracy**.
- **Saved artifacts:** `models/hy_binary_pipeline.joblib`, `models/hy_multiclass_pipeline.joblib`.
- **Welder output:** `Pred_Stage`, `P_Stage*`, `PD_Severity_Score` from the multiclass pipeline.

---

## Notebooks (optional)

- `notebooks/phase1_pd_hy_model.ipynb` — mirrors Phase 1; saves legacy `pd_hy_*.pkl` under `../models/` if run from `notebooks/`.
- `notebooks/phase2_welder_projection.ipynb` — loads `../models/hy_*_pipeline.joblib` when present.

---

## Limitations

| Issue | Impact |
|-------|--------|
| n = 14 PD, 16 welders | Wide CIs; exploratory only |
| Age gap (PD vs welders) | Major confound |
| Three balance scales only | No ABC / TUG |
| Welder labels | PD-like resemblance, not diagnosis |

---

## Development

**CI:** A workflow file is provided at `.github/workflows/ci.yml` (runs `python -m src.smoke_test` on push/PR). It must be pushed with a GitHub token that has the **`workflow` scope** (or add the file via the GitHub web UI). Requires `data/PD_WELDERS RAW Long Data-2.xlsx` in the repo for the job to pass.

```bash
python -m src.train_hy_model --help
python -m src.benchmark_group_discrimination --help
python -m src.project_welders --help
python -m src.predict_excel --help
python -m src.generate_paper_figures --help
python -m src.validate_external_pd --help
python -m src.write_summary_report --help
python -m src.smoke_test --help
```

---

*LOOCV on PD; bootstrap CIs on LOOCV vectors. Spearman for exploratory exposure associations.*
