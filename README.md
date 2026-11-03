# CSE382 - City Hotel Booking Demand
# اقسم بالله سوف اعطيكو درجة نهائية

Repository: https://github.com/Omar-Medhat1/CSE382-Hotel-Booking

## Current status

Milestone 1 technical work is implemented and executed: source verification, EDA, chronological holdouts, leakage-aware feature policies, preprocessing pipelines, untuned supervised/clustering baselines, and measured results. A separate Milestone 1 progress report is omitted following the TA's updated submission guidance. **All six team names and student IDs are recorded. Other members' GitHub accounts and actual contributions remain pending; complete those details before submitting.**

Supervised predictions apply to bookings active at the **end of the day before scheduled arrival**, reflecting the source paper's snapshot timing. The cancellation task predicts late cancellation/no-show in that risk set; it is not a booking-creation forecast. Final supervised test evaluation remains reserved for Milestone 2.

The course brief gives the Milestone 1 deadline as 5 November 2026.

## Environment setup

Use Python 3.12 (verified on 3.12.14). From the project directory on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py download
.\.venv\Scripts\python.exe main.py baseline
.\.venv\Scripts\python.exe main.py verify
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On macOS/Linux use `python3.12 -m venv .venv` and replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. Results can differ slightly across platforms/libraries; fit times are machine-dependent. Package versions are pinned in requirements.txt, randomness uses seed 382, and numerical libraries are limited to two threads during the experiments.

## Data

Use H2.csv only (City Hotel). Do not merge H1 and H2.

The `download` command fetches the CSV from the source mirror documented in the course brief:

https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv

If downloading manually, save it as `data/raw/H2.csv`, then run:

```powershell
.\.venv\Scripts\python.exe main.py validate-data
```

The validator checks the 31 expected source columns, record width, 79,330 expected records, and the verified SHA-256. Loading additionally checks dates, label/status agreement, and target domains. Original source labels are preserved. Padded categorical values are stripped; true unknowns are distinguished from Agent/Company NULL (not applicable).

Verified SHA-256: `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`.

Dataset reference: Nuno Antonio, Ana de Almeida, and Luis Nunes, "Hotel booking demand datasets," Data in Brief 22 (2019), 41-49. https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/

## Layout

```text
main.py                 Command-line entry point
config.json             Data source, schema, and experiment seed
src/data/               Source-data loading and validation
src/analysis/           Training EDA and source structural checks
src/preprocessing/      Reusable training-fitted preprocessing
src/features/           Feature allowlists and stateless engineering
src/regression/         ADR baseline
src/classification/     Late-cancellation baseline
src/clustering/         Independent full-H2 unsupervised baseline
src/evaluation/         Validation metrics
src/experiments.py      Reproduction workflow
src/inference.py        Saved-pipeline prediction
src/verification.py     Split/statistic/artifact integrity checks
data/raw/               Downloaded H2.csv (excluded from Git)
artifacts/              Generated fitted objects (excluded from Git)
results/                Figures and measured results
docs/                   Team details and implementation plan
examples/               Example booking JSON for saved inference
scripts/                Submission packaging
tests/                  Integrity tests
```

## Measured preliminary results

| Task / baseline | Evaluation | Result |
| --- | --- | --- |
| ADR / Ridge | Validation, 6,545 eligible records | MAE 17.923; RMSE 24.167; R-squared 0.541 |
| ADR / training mean | Same validation records | MAE 25.614; RMSE 35.696 |
| Cancellation / logistic regression | Same validation records | ROC-AUC 0.7043; average precision 0.06665; recall 0.00588 at threshold 0.5 |
| Cancellation / prior | Same validation records | Accuracy 0.9740; ROC-AUC 0.5; recall 0 |
| K-Means, k=3 | Full H2; silhouette sample 2,000 | Silhouette 0.2197; Davies-Bouldin 1.7386 |

**The classifier catches only 1 of 170 validation positives at the default threshold and is not operationally useful at that threshold.** High accuracy is misleading in this late-cancellation risk set. These are untuned preliminary results; no test performance is reported.

Training: 28,952 eligible arrivals before 1 January 2017; validation: 6,545 arrivals in January-April 2017; final test: 8,576 arrivals in May-August 2017. The risk-set filter excludes 34,740 records; label-availability purging excludes another 247 training and 270 validation records. See docs/leakage-audit.md and results/split_metadata.json for exact cutoff dates and limitations.

Numeric imputation/scaling and categorical encoding are fitted on training rows only. Identical predictor groups have zero overlap across chronological partitions. Repeated source records are retained because there are no unique booking IDs. Clustering uses separate full-source preprocessing; its labels and transforms never enter supervised models.

## Results and supporting notes

- [Complete preliminary leakage audit: all 31 columns](docs/leakage-audit.md)
- [Milestone 2 plan](docs/milestone-2-plan.md)
- `results/tables/`: schema, training ranges/cardinality/outlier summaries, validation metrics, cluster profiles, and compressed partition/cluster assignment CSVs.
- `results/figures/`: training EDA, validation prediction/ROC/PR/confusion plots, and cluster visualization.
- `results/run_summary.json`: exact measured results, model parameters, source fingerprint, versions, and fit times.
- `results/verification.json`: split reproduction, train-only imputation, zero predictor-group overlap, and saved-artifact checks.

`main.py analyze` can regenerate EDA without training. `main.py baseline` regenerates all analysis and baselines. The cluster PCA plot is for visualization only; supervised PCA/LDA comparisons remain Milestone 2 work.

## Saved inference

After `baseline` has generated the local artifacts:

```powershell
.\.venv\Scripts\python.exe main.py predict --task regression --input examples/booking.json
.\.venv\Scripts\python.exe main.py predict --task classification --input examples/booking.json
```

Inputs use the original booking source names. The example is a synthetic active booking; the caller must ensure the booking is still active at the prediction point. Inference loads saved pipelines and does not retrain. Unconstrained Ridge may produce negative rates on unfamiliar inputs; it is a baseline, not a production pricing tool. Fitted artifacts remain local and are excluded from Git and submission ZIPs.

## Team and contributions

| Name | Student ID | Assigned responsibility |
| --- | --- | --- |
| OMAR MEDHAT | 23P0026 | Preprocessing, temporal splits and leakage prevention |
| AHMED MEDHAT | 230039 | Data quality, cleaning rules, EDA and feature diagnostics |
| AHMED ADEL | 23P0046 | Regression workflow, error analysis and inference checks |
| MOHAMED HASSAN | 23P0042 | Classification workflow, imbalance analysis and inference checks |
| AHMED MOKHTAR | 23P0081 | Clustering workflow, PCA analysis and segment interpretation |
| HALA ELTAWEEL | 23P0232 | CLI integration, system checks and reproducible submission |

The confirmed team has six members; see `docs/team.md` and `docs/team.json`. [The work allocation](docs/contributions.md) gives each member a complete work package with technical ownership, analysis, and verification. Hala coordinates CLI/system integration, checks setup/run instructions, and verifies the submission package. Omar retains repository-owner access administration. Roles are assigned; completed contributions and other members' GitHub usernames remain pending. Each member must contribute through their own GitHub account and record actual work and verification evidence.

## Submission

Complete the pending GitHub account/contribution details and commit the final version before submitting. The package includes source code, configuration, README, dependencies, measured result tables/figures, and supporting audit/team notes. The separate Milestone 1 report is omitted under the TA's updated guidance. The package also excludes the downloaded source CSV, trained models/fitted pipelines, virtual environments, secrets, and Git internals; the data and artifacts can be regenerated with the exact commands above.

From a clean committed checkout, create the ZIP:

```powershell
.\.venv\Scripts\python.exe scripts/package_submission.py --output ../Milestone1_Draft.zip
```

The packager fills the submitted commit SHA in the archive README and adds SUBMISSION.json with file checksums. This archive-only metadata substitution avoids a self-referential Git commit hash; it does not modify the source checkout. The online commit history can be reviewed through the repository URL above.

Submitted commit SHA: __SUBMITTED_COMMIT_SHA__

The placeholder is automatically replaced in the packaged README. The roster is confirmed; this remains a draft until other GitHub accounts and actual contribution records are complete.
