# Model card — PD-referenced motor balance (pilot)

## Intended use

- **Primary:** Train **PD-only** classifiers from **BBS, Mini-BEST, and FES** to approximate **Hoehn & Yahr** severity (binary Early/Late and multiclass stages I–IV), then apply the **saved multiclass pipeline** to **welder** (or other) rows that have the same three features.
- **Outputs** are **PD-like motor-balance resemblance** measures: predicted stage, stage probabilities, and a probability-weighted **PD-like severity score** (not a clinical H&Y assignment outside the PD training distribution).

## Not intended for

- **Diagnosis** of Parkinson’s disease in individuals.
- **Clinical staging** of welders or other occupational groups as true H&Y.
- **Treatment decisions** or screening without independent validation and clinical oversight.

## Inputs

| Concept | Role |
|--------|------|
| **BBS** | Berg Balance Scale (numeric) |
| **Mini-BEST** | Mini-BESTest total (numeric; column name may vary — auto-detected) |
| **FES** | Falls Efficacy Scale (numeric) |
| **Row ID** | Optional identifier column (e.g. `ID`) |

Training expects a **PD** sheet with **true H&Y** labels; inference on new sheets does not require H&Y.

## Outputs (typical)

| Field | Meaning |
|-------|---------|
| `Pred_Stage` | Argmax stage from multiclass probabilities |
| `P_Stage1` … `P_Stage4` | Predicted probability per trained stage |
| `PD_Severity_Score` | Σ *k* · P(stage *k*) — continuous PD-like severity in 1–4 space |
| `Pred_Binary` / labels | Optional Early-like vs Late-like from binary pipeline |

## Limitations

- **Small PD sample** (*n* = 14 in the pilot Excel); wide uncertainty; LOOCV is optimistic for generalization.
- **Class imbalance** in H&Y; multiclass exact accuracy is modest; **within-one-stage** accuracy is reported separately.
- **Confounding** (e.g. age) between PD and welder cohorts; welder outputs are **not** externally validated against true H&Y.
- **Three scales only** — not a full motor or clinical assessment.

## External validation status

- **Not established** on an independent PD cohort in this repository’s primary results.
- For protocol when independent PD data exist: **`docs/external_validation.md`** and `python -m src.validate_external_pd`.
- Welders **cannot** externally validate H&Y accuracy (no gold-standard H&Y).

## Provenance

- Selection rule for saved pipelines: **`docs/model_design.md`** (Combined features; ordered criteria for binary vs multiclass).
- Reproducibility: random seed and dependencies in **`src/utils.py`** and **`requirements.txt`**.
