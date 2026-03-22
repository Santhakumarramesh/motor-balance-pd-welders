# External validation (design and paper wording)

## Why it matters

Internal validation (leave-one-out CV on the development PD cohort) fits a small pilot but does not answer whether the **frozen** PD reference model generalizes. **External validation** means: train (or fix) the model using the original PD data, then evaluate it on a **separate PD dataset** that was **not** used for fitting or model selection, with **true H&Y** labels and the same balance measures.

Welders **cannot** substitute for external PD validation: they have **no** true H&Y stage.

## What to do when external data are not yet available

Use clear, reviewer-friendly language in the **Limitations** or **Discussion**:

> External validation was not performed in this pilot study. The PD reference models were assessed with leave-one-out cross-validation on a single small cohort (*n* = 14 PD patients). Findings are therefore hypothesis-generating and require evaluation on an **independent PD sample** with H&Y staging and the same balance instruments before broader interpretation or clinical use.

Optional **next step**:

> Future work should apply the saved multiclass and binary pipelines, **without retraining**, to an independent PD cohort reporting BBS, Mini-BESTest, and FES, and report accuracy, balanced accuracy, macro-F1, within-one-stage accuracy, and confusion matrices.

## Target design (when you have data)

1. **Freeze** the pipelines saved by `python -m src.train_hy_model` (`hy_binary_pipeline.joblib`, `hy_multiclass_pipeline.joblib`).
2. Obtain a **new** PD dataset (ideally another site or time window), with:
   - Clinically assigned H&Y stage  
   - BBS, Mini-BEST, FES (same instruments / comparable scoring)
3. **Do not retrain** on the external cohort for the primary external-validation estimate (optional sensitivity: retrain on combined data in a separate experiment).
4. Run:

   ```bash
   python -m src.validate_external_pd path/to/external_pd.xlsx --sheet PD -o outputs/metrics/external_validation.json
   ```

5. **Report** for binary and multiclass:
   - Accuracy, balanced accuracy, macro-F1  
   - Multiclass: within-one-stage accuracy  
   - Confusion matrices  
   - `classification_report` strings (already saved in JSON)

## Partial overlap (two of three scales)

If the external cohort only has two measures (e.g. BBS + FES), you cannot use the full three-feature pipeline fairly. Options:

- Train a **separate** reduced-feature model on the **development** PD data only, freeze it, then validate that reduced model externally; or  
- Impute missing scale (weak) or report as **secondary** analysis with explicit limitations.

State limitations transparently.

## Calibration (optional extensions)

For publication you may add probability calibration (e.g. reliability diagrams, Brier score) on the external PD cohort using **multiclass** predicted probabilities. Not implemented in the default JSON export; extend `validate_external_pd.py` if needed.

## Warning

Running `validate_external_pd.py` on the **same** Excel file used to **train** the pipelines is **not** external validation: the saved models were fit on that PD sheet, so metrics will be **overly optimistic** (often near-perfect on the training distribution). Use a **different** PD file for a legitimate external check.

## Summary

| Question | Answer |
|----------|--------|
| Can welders validate H&Y accuracy? | **No** — no true H&Y labels. |
| What validates the PD model? | **Independent PD cohort** with true H&Y + same features. |
| What does this repo provide now? | LOOCV internally + **script** for frozen-model external PD evaluation when data exist + honest wording if data do not. |
