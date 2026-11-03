# Work allocation and contribution record

Status: responsibilities are assigned across six members. Record completed work and evidence as each member performs, checks, and improves their component. Individual completion has not yet been verified.

## Assigned responsibilities

| Member | Student ID | Assigned area | Evidence to produce |
| --- | --- | --- | --- |
| OMAR MEDHAT | 23P0026 | Preprocessing, feature engineering, temporal splitting, leakage audit; GitHub coordination | Feature/pipeline review; split and leakage checks; justified corrections and integration |
| AHMED MEDHAT | 230039 | Dataset validation and EDA | Source-check and EDA review; documented findings; links to actual changes |
| AHMED ADEL | 23P0046 | ADR regression and evaluation | Reproduced Ridge/reference metrics; residual analysis; model limitations |
| MOHAMED HASSAN | 23P0042 | Cancellation classification and evaluation | Reproduced classification metrics; confusion/ROC/PR interpretation; imbalance analysis |
| AHMED MOKHTAR | 23P0081 | Clustering, profiles, PCA visualization | Reproduced cluster metrics and profiles; interpretation and geometry limitations |
| HALA ELTAWEEL | 23P0232 | Report, reproducibility, submission packaging | Report checked against outputs; setup/inference/package checks; final draft review |

## Concrete tasks and completion criteria

### AHMED MEDHAT - data and EDA

- Review `src/data/loading.py`, `src/data/validation.py`, and `src/analysis/eda.py`.
- Run `python main.py download`, `python main.py validate-data`, and `python main.py analyze` using the project environment.
- Check the 79,330-by-31 schema, fingerprint, padded source strings, four unknown Children values, 24 unknown Country values, and Agent/Company not-applicable categories.
- Review duplicate handling, unusual values, category cardinality, distributions, and training-only plots. Document findings and make any justified corrections.
- Completion evidence: command results, an EDA explanation, and links to the member's actual changes or substantive review.

### OMAR MEDHAT - preprocessing, leakage prevention and GitHub coordination

- Review `src/features/policy.py`, `src/preprocessing/pipelines.py`, `src/data/splitting.py`, and `docs/leakage-audit.md`.
- Explain the day-before-arrival prediction point, active-booking risk set, label-availability purging, and all feature exclusions.
- Check median/UNKNOWN imputation, categorical IDs, one-hot handling of unseen categories, scaling, and deterministic stay/calendar features.
- Run `python -m unittest discover -s tests -v` and, after baselines exist, `python main.py verify`. Check training-only fitted statistics and zero cross-partition predictor-group overlap.
- Document any timing assumptions or limitations and make evidence-backed corrections without consulting final-test performance.
- Maintain repository setup instructions, add teammates as collaborators once their usernames are supplied, and coordinate integration of their actual changes.
- Completion evidence: feature/pipeline audit, verified checks, and links to actual changes or substantive review.

### AHMED ADEL - regression

- Review `src/regression/baseline.py`, regression metrics in `src/evaluation/metrics.py`, and the regression branch of `src/experiments.py`.
- Reproduce the baseline workflow with `python main.py baseline` and examine Ridge versus the training-mean reference in `results/tables/supervised_validation_metrics.csv`.
- Explain MAE, MSE/RMSE, R-squared, alpha=1, scaling, categorical encoding, and the validation residual/prediction pattern.
- Check saved regression inference with `python main.py predict --task regression --input examples/booking.json`.
- Document important errors and negative-prediction limitations; propose regression models and a training-fold tuning plan for Milestone 2.
- Completion evidence: regression result review, an explanation of errors/metrics, and links to actual changes or substantive review.

### MOHAMED HASSAN - classification

- Review `src/classification/baseline.py`, classification metrics in `src/evaluation/metrics.py`, and classification plots in `src/experiments.py`.
- Reproduce the baseline workflow and verify the prior and logistic-regression results, positive prevalence, and confusion matrix.
- Explain accuracy, precision, recall, F1, ROC-AUC, average precision, and why the 0.5 threshold catches only 1 of 170 validation positives.
- Check saved classification inference with `python main.py predict --task classification --input examples/booking.json`.
- Document the poor default-threshold recall. Plan class weighting, calibration, and threshold evaluation using training-fold predictions for Milestone 2.
- Completion evidence: classification result review, imbalance/threshold explanation, and links to actual changes or substantive review.

### AHMED MOKHTAR - clustering

- Review `src/clustering/baseline.py`, the clustering feature allowlist, and clustering analysis in `src/experiments.py`.
- Reproduce the baseline workflow and inspect `clustering_metrics.csv`, `cluster_profiles.csv`, and the cluster/PCA figure.
- Explain k=3 as an initial baseline, log transforms, scaling, one-hot categories, sampled silhouette, Davies-Bouldin, and the roughly 35% variance retained in the 2-D visualization.
- Interpret each cluster using booking attributes. Investigate sensitivity to rare-count weighting as planned work; keep outcome labels out of clustering decisions.
- Prepare the plan for optimized K-Means, hierarchical clustering, DBSCAN, and stability checks in Milestone 2.
- Completion evidence: cluster-profile explanation, metric/projection limitations, and links to actual changes or substantive review.

### HALA ELTAWEEL - report and reproducibility

- Review `src/reporting.py`, `README.md`, `scripts/package_submission.py`, and the report against the measured tables and figures.
- Follow the README setup/reproduction commands, check the six names and IDs, and verify that validation metrics are clearly separated from reserved test evaluation.
- Run `python main.py report`, `python main.py verify`, and both example inference commands after baselines exist; inspect the generated PDF pages for layout and accuracy.
- Check source references, limitations, remaining-work plan, and the contribution evidence supplied by the other members.
- After final changes are committed, run `python scripts/package_submission.py --output ../Milestone1_Draft.zip`; check the recorded SHA and exclusion of downloaded data, environments, and fitted models.
- Completion evidence: report corrections, reproducibility/package review, and links to actual changes or substantive review.

## How each member records actual work

1. Use the member's own GitHub account and review the assigned area before changing it.
2. Run the relevant commands, record findings, and make justified code, analysis, documentation, or review contributions.
3. Commit actual changes, including documented findings where appropriate, under that member's account. Also link substantive reviews explaining what was checked and found. Preserve the real development history.
4. Add the date, completed work, commit/review link, and verification result below. Mark completion only after the work is done.

All commands above run from the repository root using the project environment. On Windows, replace `python` with `.\.venv\Scripts\python.exe`.

## Completed contribution log

| Date | Member | Actual work completed | Commit or pull request | Verification |
| --- | --- | --- | --- | --- |

Completed entries are pending. Every member should also understand the complete workflow and be able to explain its assumptions and results. Commit authorship alone does not document technical understanding.
