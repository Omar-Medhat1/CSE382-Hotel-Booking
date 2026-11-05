# Mohamed Hassan - classification workflow and imbalance review

- Student ID: 23P0042
- Review date: 5 November 2026
- Working branch: `mohamed-classification-review`
- Source commit reviewed: `caf56437d12f77a9703d22da5d1ddc478b6b9c0e`

## Scope checked

The classification target is class 1, a late cancellation or no-show among bookings that were still active at the end of the day before scheduled arrival. This is a restricted, reconstructed risk set, not a booking-creation forecast and not a model for every source cancellation. Eligibility requires `LeadTime >= 1` and a final status date strictly after the snapshot date. The temporal split and label-availability purge remain unchanged, and no final-test labels were evaluated.

The untuned logistic-regression pipeline applies training-fitted median imputation and scaling to numeric inputs, training-fitted imputation and one-hot encoding to categorical inputs, then fits `LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=382, class_weight=None)`. The fitted pipeline is saved to `artifacts/classification_baseline.joblib`, reloaded for verification and CLI inference, and converts class-1 probabilities to decisions at the fixed threshold 0.5.

## Commands actually run and outcomes

All commands ran from the repository root with Python 3.12.14 and the pinned packages in `requirements.txt`.

```text
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Result after the classification changes: 11 tests passed.

```text
.\.venv\Scripts\python.exe main.py baseline
```

Result: completed successfully using the verified 79,330-row H2 source with SHA-256 `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`. It reproduced 28,952 training rows, 6,545 validation rows, 170 validation positives, and the previously recorded classification metrics. The reserved final test was not evaluated. Machine-dependent fit-time changes were not retained.

```text
.\.venv\Scripts\python.exe main.py verify
```

Result: passed source checksum, split reproduction, zero predictor-group overlap, train-only imputation, artifact reload probabilities, class ordering `[0, 1]`, probability/decision consistency, and probability-summary reproduction. `final_test_evaluated` remained `false`.

```text
.\.venv\Scripts\python.exe main.py predict --task classification --input examples\booking.json
```

Result: class-1 probability `0.08652896369415897`, decision `0` at threshold 0.5.

Additional integration cases were run from temporary copies of the example input:

- Changing `Country` to `UNSEEN_COUNTRY` succeeded (exit 0), with probability `0.030026818492705935` and decision `0`; the fitted one-hot encoder safely ignores unseen categories.
- Removing the required `Country` key failed as intended (exit 1): `Missing keys: ['Country']; unexpected keys: []`.

## Validation findings

The training-prior reference predicts no positives. Its validation accuracy is `0.974026`, recall is `0`, ROC-AUC is `0.5`, and average precision is the validation prevalence, `0.025974`.

The untuned logistic model reproduced:

- Confusion matrix in class order `[0, 1]`: `[[6366, 9], [169, 1]]`.
- Accuracy `0.972804`, precision `0.100000`, recall `0.005882`, and F1 `0.011111`.
- ROC-AUC `0.704285` and average precision `0.066650`.

High accuracy is misleading because 6,375 of 6,545 validation rows are negatives. Predicting every row as negative is therefore 97.40% accurate. At threshold 0.5, logistic regression produces 169 false negatives, 9 false positives, and only 1 true positive. Precision is `1 / (1 + 9) = 0.10`; recall is `1 / 170 = 0.005882`.

ROC-AUC measures pairwise ranking across all possible thresholds, so it can exceed 0.7 even when the single 0.5 operating point has almost zero recall. Average precision also evaluates the ranking, but it emphasizes the rare positive class and is only `0.066650`: better than the `0.025974` prevalence reference, but still weak. Scikit-learn average precision is a recall-increment-weighted summary of the precision-recall curve; it should not be described as a casually computed trapezoidal PR area.

The class-wise validation probability summary adds the following evidence:

| True class | Rows | Mean | Median | P95 | Maximum | Decisions at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Not canceled (0) | 6,375 | 0.027783 | 0.013878 | 0.104061 | 0.958650 | 9 |
| Late cancel/no-show (1) | 170 | 0.062417 | 0.023426 | 0.229548 | 0.675687 | 1 |

The positive distribution is shifted upward, which agrees with the above-chance ranking metrics, but the classes overlap heavily and almost all positive probabilities remain below 0.5. This is descriptive validation analysis, not a new selected threshold.

## Changes made

- `src/evaluation/metrics.py`: centralized validated probability-to-decision conversion and added a reproducible class-wise probability summary.
- `src/experiments.py`: verifies class order `[0, 1]`, exports the probability summary, and creates a class-normalized log-scale probability-distribution figure.
- `src/inference.py`: locates class 1 from the fitted artifact instead of silently assuming a probability column.
- `src/verification.py`: checks class order, probability rows summing to one, model-predict/0.5-decision agreement, reload probabilities, and the exported probability summary.
- `tests/test_classification_diagnostics.py`: covers positive-class thresholding, confusion-matrix orientation, summary grouping, and invalid probabilities.
- `results/tables/classification_validation_probability_summary.csv`: measured class-wise probability diagnostics.
- `results/figures/classification_validation_probability_distribution.png`: validation probability distributions by true class.
- `results/verification.json`: recorded successful classification consistency checks.

## Limitations and Milestone 2 plan

The baseline remains deliberately untuned. Validation results were not used to claim a new operating threshold, and the reserved final test remains untouched. The source lacks booking/customer identifiers, and the reconstructed prior-day risk set depends on date-only timing assumptions.

For Milestone 2, generate out-of-fold probabilities using only the training cohort. Within that training-only process, compare class weighting, evaluate probability calibration, and select a threshold using a stated cost or recall/precision objective. Any calibration method and operating threshold must be fitted or selected from training folds, then frozen before validation comparison and the one-time final test evaluation. Report ranking, calibration, and thresholded decision metrics separately.

## Short discussion explanation

The classifier ranks late cancellations better than chance, but class 1 is only about 2.6% of validation rows and the untuned probabilities are mostly below 0.5. That is why ROC-AUC is about 0.70 while recall at the fixed threshold is only 1 out of 170. Accuracy stays near 97% mainly because negatives dominate. I verified that class 1 is probability column 1, that saved and reloaded probabilities match, and that CLI decisions use the same 0.5 rule. Threshold, weighting, and calibration choices belong in a training-fold procedure for Milestone 2, not in test-guided tuning.

## Draft contribution-log entry

| Date | Member | Actual work completed | Commit or pull request | Verification |
| --- | --- | --- | --- | --- |
| 2026-11-05 | MOHAMED HASSAN (23P0042) | Reproduced and reviewed the classification baseline; added rare-positive probability diagnostics, explicit positive-class and saved-decision checks, inference cases, tests, and a training-fold Milestone 2 plan. | Add the real commit/PR link after publication. | 11 tests passed; baseline reproduced; verification passed; valid and unseen-category inference succeeded; missing-key input rejected; final test not evaluated. |
