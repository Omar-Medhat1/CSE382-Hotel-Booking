# Regression baseline and error analysis — Ahmed Adel (23P0046)

**Milestone 1 | 5 November 2026**
**Branch:** `ahmed-adel-regression-review`
**Starting commit:** `36ab1e6`

## What I worked on

My part was to check the ADR regression baseline and see where its predictions go wrong. I ran the existing project, verified that the saved Ridge model loads correctly, and added a separate script to analyze the errors on the validation data.

I did not change the team's training, preprocessing, or inference code. I also kept the final test set out of this analysis.

## Reproducing the baseline

I downloaded `H2.csv` using the project command. The dataset check passed with **79,330 rows and 31 columns**. The experiment used **28,952 training rows** and **6,545 validation rows**.

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Training mean | 25.6136 | 35.6959 | -0.0004 |
| Ridge | 17.9298 | 24.1703 | 0.5413 |

Ridge had a lower error than the training-mean baseline. My numbers were very close to the results already in the repo, but not exactly the same. I haven't confirmed the reason for that small difference.

## Error analysis

I used `scripts/regression_diagnostics.py` to load the saved Ridge model and get predictions for the **validation set only**. The script did not retrain the model.

I calculated the residual as **actual ADR − predicted ADR**:

- Positive residual: the predicted value was too low.
- Negative residual: the predicted value was too high.

The average residual was **-8.94**, so the model overpredicted ADR by about **8.94 units on average**. It also gave **22 negative predictions**. I counted them but did not change them, because I wanted to check the original model as it is.

### Error for different ADR ranges

| Actual ADR | Number of bookings | MAE |
| --- | ---: | ---: |
| 0 | 85 | 71.23 |
| Above 0 to 50 | 37 | 47.34 |
| Above 50 to 100 | 4,014 | 17.08 |
| Above 100 to 150 | 1,874 | 13.62 |
| Above 150 to 200 | 413 | 24.13 |
| Above 200 | 122 | 45.04 |

The best range was **100–150 ADR**. The model had bigger errors for very low and very high rates. It generally predicted low ADR too high and high ADR too low. Some ranges have few bookings, so I wouldn't judge the whole model from those ranges alone.

### Rare and unseen categories

I also checked whether a booking contains category values that were rare or not found in the training data. I used **training data only** to decide this. Here, rare means fewer than 10 training examples for a non-missing value. Missing values were not treated as unseen automatically.

| Category group | Number of bookings | MAE |
| --- | ---: | ---: |
| No rare or unseen category | 5,937 | 17.32 |
| At least one rare category | 219 | 20.83 |
| At least one unseen category | 389 | 25.61 |

Bookings with unseen categories had higher errors. This is something worth investigating, but these results alone don't prove that the unseen categories caused the errors.

## Checks I ran

- `main.py verify` passed the data split, training-only imputation, and saved-model reload checks.
- The saved regression model predicted **119.1422 ADR** for `examples/booking.json`.
- The full unit-test suite passed: **8 tests**, including **4 new tests** for the diagnostics.
- The final test set was **not evaluated**.

## Files for my part

- `scripts/regression_diagnostics.py` — analyzes Ridge validation errors and creates the outputs.
- `tests/test_regression_diagnostics.py` — checks the error calculations and the category grouping.
- `results/tables/regression_validation_residual_summary.csv` — detailed error results.
- `results/figures/regression_validation_residuals.png` — residual plot and MAE by ADR range.
- `docs/member-notes/ahmed-adel-regression.md` — these notes.

## Commands to reproduce

Run these from the project folder in PowerShell after installing the requirements and downloading the dataset:

```powershell
.\.venv\Scripts\python.exe main.py baseline
.\.venv\Scripts\python.exe main.py verify
.\.venv\Scripts\python.exe main.py predict --task regression --input examples/booking.json
.\.venv\Scripts\python.exe -m scripts.regression_diagnostics
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## What I would check next

For Milestone 2, I want to compare Ridge with other regression models using the same split, then tune them properly. I also want to check how to handle zero-rate bookings and negative predictions without making the validation error worse. **I haven't run these experiments yet.**
