# Reviewer-oriented brief (strict framing)

This note summarizes how an independent evaluator might read this repository: **strong exploratory pilot package**, **not** a clinically validated predictive system.

## Major strengths

- **Clear primary design:** PD-only H&Y reference model on **BBS, Mini-BEST, and FES**; welder outputs framed as **PD-like resemblance**, not diagnosis — appropriate and explicit.
- **Validation matches sample size:** **LOOCV** on PD only; preprocessing inside the pipeline; bootstrap intervals on LOOCV prediction vectors — reasonable for a small pilot.
- **Model comparison:** Logistic regression, random forest, gradient boosting; production selection on **Combined** features uses **macro-F1 → balanced accuracy → (multiclass) within-one-stage accuracy → accuracy** — see **`docs/model_design.md`**.
- **Reproducibility:** Script pipeline (`run_all.py`), generic Excel inference, external-validation entry point, pinned dependencies.

## Major concerns (inherent to the study)

- **Small cohort:** 15 PD / 16 welders (default `PD_WELDERS RAW Long Data.xlsx`) limits stability, confidence intervals, and generalizability — acknowledged in-repo but remains the main scientific ceiling.
- **No completed external validation** on an independent PD cohort with true H&Y; protocol and script: **`docs/external_validation.md`**, `python -m src.validate_external_pd`.
- **PD vs welder benchmark** is **descriptive** and **confounded** (e.g. age/cohort); not disease-specific separation — see **`docs/model_design.md`** (supporting benchmark section).
- **Three balance scales only** — interpretable but limited predictive richness vs. broader clinical or gait assessment.

## Minor documentation (status)

| Concern | Status |
|--------|--------|
| README vs `model_design.md` selection wording | **Aligned** — README “Model pipeline (summary)” lists the same ordered criteria as `model_design.md`. |
| Single post-run summary | **`outputs/summary_report.md`** — generated at end of `run_all.py` (`src/write_summary_report.py`). |
| Model card | **`docs/model_card.md`** — intended use, non-diagnostic scope, I/O, limits. |

## Final recommendation

> **Accept as a strong exploratory pilot model package, but not as a clinically validated predictive system.**

Continue to present the work as **hypothesis-generating**, **PD-referenced**, **non-diagnostic**, and **pending external validation**.

**Strict verdict:** strong exploratory research model and strong repo package; **major data-related limits remain**, especially sample size and lack of independent validation.
