# Motor Balance Analysis: PD-Referenced Severity Model + Welder Projection

> **Pilot study — exploratory analysis only.**  
> n = 30 (14 PD, 16 welders). Hypothesis-generating, not clinically validated.

**Repository:** [github.com/Santhakumarramesh/motor-balance-pd-welders](https://github.com/Santhakumarramesh/motor-balance-pd-welders)

---

## Primary research design

**Main claim:** Train a **PD-only** H&Y reference model on **BBS, Mini-BEST, and FES** (leave-one-out CV), save the best pipelines, then **classify welder rows into PD-like motor-balance severity categories** using multiclass stage probabilities and a **PD-like severity score**. This is **resemblance in a PD-learned space**, not a diagnosis of PD in welders and not a true clinical H&Y stage in welders.

**Secondary (optional):** Correlations between PD-like severity and welding exposure variables are **exploratory**; see `outputs/metrics/phase2_associations.json`. Do not treat them as the headline finding.

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

1. `python -m src.train_hy_model` — PD LOOCV, feature ablation, saves `models/hy_*_pipeline.joblib`, `outputs/metrics/phase1_metrics.json`, `outputs/figures/fig_01`–`fig_05`.
2. `python -m src.project_welders` — welder projection, `outputs/predictions/welder_predictions.xlsx`, `fig_06`–`fig_07`.
3. `python -m src.generate_paper_figures` — paper-ordered figures + captions: `outputs/figures/paper/`.

**Data:** `data/PD_WELDERS RAW Long Data-2.xlsx`  
**Seed:** `42` (`src/utils.py`)

---

## Other entry points

| Command | Purpose |
|---------|---------|
| `python -m src.predict_excel INPUT.xlsx` | Inference on **any** Excel with BBS / Mini-BEST / FES columns (auto-detected); writes `outputs/predictions/inference_predictions.xlsx` by default |
| `python -m src.generate_paper_figures --ensure-run` | Build paper figures + `.md` captions; if needed, runs full pipeline first |

Design reference: **`docs/model_design.md`**.  
Narrative summary: **`results.txt`** (align numbers with `phase1_metrics.json`).

---

## Repository layout

```
motor-balance-pd-welders/
├── data/
│   └── PD_WELDERS RAW Long Data-2.xlsx
├── src/
│   ├── utils.py
│   ├── train_hy_model.py
│   ├── project_welders.py
│   ├── predict_excel.py
│   └── generate_paper_figures.py
├── models/                   # Generated *.joblib (gitignored)
├── outputs/
│   ├── figures/              # Pipeline fig_01 … fig_07
│   ├── figures/paper/        # paper_fig_*.png + .md captions
│   ├── metrics/              # phase1_metrics.json, phase2_associations.json
│   └── predictions/
├── notebooks/
├── docs/
│   ├── model_design.md
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
- **Selection:** Best **macro-F1** on the **Combined** feature set among logistic regression, random forest, gradient boosting; saved pipelines in `models/`.
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

```bash
python -m src.train_hy_model --help
python -m src.project_welders --help
python -m src.predict_excel --help
python -m src.generate_paper_figures --help
```

---

*LOOCV on PD; bootstrap CIs on LOOCV vectors. Spearman for exploratory exposure associations.*
