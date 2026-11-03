# Milestone 1 implementation plan

This document records the implementation plan. Executed evidence and current decisions are in README.md, docs/leakage-audit.md, and results/run_summary.json. Milestone 1 is submitted without a separate progress report following the TA's updated guidance. Source investigation established a day-before-arrival prediction point and a late-cancellation risk set. Remaining final-project experiments are documented in docs/milestone-2-plan.md.

## Data understanding

Preserve the original H2 source. Record its checksum, dimensions, types, missing-value tokens, duplicate rows, cardinality, and data ranges. Investigate zero-night and zero-guest bookings, extreme ADR, negative rates, and unusual ages or counts where relevant. Do not drop repeated bookings solely because feature rows match; establish and document the duplicate policy.

## Prediction points and leakage

Define separate feature policies for ADR regression, eventual cancellation classification, and unsupervised booking segmentation. Resolve what information is known at each prediction point before selecting features. Examine final reservation status, status dates, eventual room assignment, booking changes, and the timing of deposits and waiting-list values. Exclude each supervised target from its own inputs and avoid using outcomes to define or tune clusters.

## Experimental protocol

Use seed 382 wherever randomness is involved. Choose and justify a chronological or otherwise appropriate split after inspecting date coverage and duplicate/group relationships. Keep the final supervised test partition unavailable to model-selection code. Fit imputers, encoders, scalers, and any learned feature transformations on training data only. Report validation performance for Milestone 1.

## Initial model candidates

- Regression: Ridge regression with an appropriate numeric/categorical preprocessing pipeline; report MAE, MSE/RMSE, and R-squared.
- Classification: logistic regression; report accuracy, precision, recall, F1, confusion matrix, and ROC-AUC and/or PR-AUC.
- Clustering: K-Means on a justified, scaled booking representation; report cluster sizes, sampled silhouette if needed, Davies-Bouldin index, and descriptive profiles. Labels must not guide cluster selection.

These are candidate choices pending data inspection. Run untuned baselines for Milestone 1. Document all defaults and runtime versions. Do not present proposed or invented numbers as measured results.

## Submission and remaining project

Submit the reproducible code, setup/run instructions, result tables and figures, and supporting audit/team notes. Record actual member contributions as the work is performed. No separate progress report is included for Milestone 1. Milestone 2 still requires the final project report and at least three model families per task with tuning, PCA/LDA comparisons, final supervised evaluation, saved inference artifacts, and the final application.
