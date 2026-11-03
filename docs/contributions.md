# Work allocation and contribution record

Status: responsibilities are assigned across six members. Record completed work and evidence as each member performs, checks, and improves their component. Individual completion has not yet been verified.

## Assigned responsibilities

| Member | Student ID | Assigned area | Evidence to produce |
| --- | --- | --- | --- |
| OMAR MEDHAT | 23P0026 | Preprocessing, temporal splits, leakage prevention | Reusable transformations; split/fit-isolation checks; feature and leakage audit |
| AHMED MEDHAT | 230039 | Data quality, cleaning rules, EDA, feature diagnostics | Data-quality contract; verified EDA outputs and interpretations |
| AHMED ADEL | 23P0046 | Complete regression workflow, error analysis, inference checks | Model/reference comparison; residual analysis; saved prediction checks |
| MOHAMED HASSAN | 23P0042 | Complete classification workflow, imbalance analysis, inference checks | Metrics/curves; rare-positive analysis; probability/decision checks |
| AHMED MOKHTAR | 23P0081 | Complete clustering workflow, PCA analysis, segment interpretation | Unsupervised representation; metrics/profiles/projection; segment interpretation |
| HALA ELTAWEEL | 23P0232 | CLI integration, system checks, reproducible submission | Working command flows; environment/reload/package checks; setup/run instructions |

## Comparable work packages and handoffs

Each member owns three deliverables: their technical component, evidence-backed analysis, and relevant verification. Each should also be able to walk through the component. No separate Milestone 1 report or individual report section is assigned under the TA's updated guidance. Workloads are comparable in scope; actual effort depends on the problems found and should be reviewed after the first handoff. Counting model constructor lines or commits is not a useful measure of effort.

- Ahmed Medhat owns source parsing/normalization and the data-quality contract. Omar owns learned preprocessing, deterministic feature engineering, and split isolation. Agree on missingness/category meanings before changing either component.
- Omar defines shared feature/split rules. Regression, classification, and clustering owners explain their own task-specific inputs and limitations and contribute to the audit.
- Ahmed Adel and Mohamed Hassan own their respective metric functions and experiment branches. They supply task-specific prediction cases to Hala, who owns common CLI dispatch, input handling, and integration. Coordinate edits to shared files such as `src/experiments.py`, `src/evaluation/metrics.py`, and `src/inference.py`.
- Ahmed Mokhtar owns the independent clustering representation, fitted clustering pipeline, evaluation, profiles, and PCA display. Supervised PCA/LDA comparisons remain planned Milestone 2 work.
- Each technical owner checks their result tables, figures, and interpretation. Hala checks that outputs, README commands, shared interfaces, and the packaged project are consistent.
- Hala coordinates technical integration and end-to-end checks. Omar retains repository-owner access administration, such as adding collaborators when usernames are supplied.
- Verification must address actual risks or uncovered behavior. Reuse existing checks where appropriate; add meaningful tests when a real coverage gap or fix warrants them. Avoid cosmetic changes solely to create commits.

## Concrete tasks and completion criteria

### AHMED MEDHAT - data quality, cleaning and EDA

- Review `src/data/loading.py`, `src/data/validation.py`, and `src/analysis/eda.py`.
- Run `python main.py download`, `python main.py validate-data`, and `python main.py analyze` using the project environment.
- Check the 79,330-by-31 schema, fingerprint, padded source strings, four unknown Children values, 24 unknown Country values, and Agent/Company not-applicable categories.
- Review duplicate handling, unusual values, category cardinality, distributions, and training-only plots. Document findings and make any justified corrections.
- Own the source-data quality contract: required fields/types, allowed missing-value tokens, not-applicable IDs, date/domain checks, and the documented policy for unusual values. Coordinate with Omar so source normalization and fitted imputation stay consistent.
- Check that malformed or incompatible source data is rejected clearly, and verify that descriptive conclusions match exported tables. Inspect relevant training feature associations and category concentration without examining final-test target performance.
- Verify EDA figures, unusual-value decisions, and feature-distribution limitations, and explain the resulting findings during the component walkthrough.
- Completion evidence: verified parsing/quality behavior, checked EDA outputs and interpretations, and links to actual changes or substantive review.

### OMAR MEDHAT - preprocessing, temporal splitting and leakage prevention

- Review `src/features/policy.py`, `src/preprocessing/pipelines.py`, `src/data/splitting.py`, and `docs/leakage-audit.md`.
- Explain the day-before-arrival prediction point, active-booking risk set, label-availability purging, and all feature exclusions.
- Check median/UNKNOWN imputation, categorical IDs, one-hot handling of unseen categories, scaling, and deterministic stay/calendar features.
- Run `python -m unittest discover -s tests -v` and, after baselines exist, `python main.py verify`. Check training-only fitted statistics and zero cross-partition predictor-group overlap.
- Document any timing assumptions or limitations and make evidence-backed corrections without consulting final-test performance.
- Maintain the standalone feature/leakage audit and obtain task-specific timing/input explanations from the three model owners.
- Hand verified input schemas and split contracts to the model owners and Hala. Hala coordinates technical integration; Omar handles repository-owner collaborator access when usernames are supplied.
- Completion evidence: reusable transforms, split/fit-isolation checks, the audit, and links to actual changes or substantive review.

### AHMED ADEL - complete regression workflow and diagnostics

- Review `src/regression/baseline.py`, regression metrics in `src/evaluation/metrics.py`, and the regression branch of `src/experiments.py`.
- Reproduce the baseline workflow with `python main.py baseline` and examine Ridge versus the training-mean reference in `results/tables/supervised_validation_metrics.csv`.
- Explain MAE, MSE/RMSE, R-squared, alpha=1, scaling, categorical encoding, and the validation residual/prediction pattern.
- Check saved regression inference with `python main.py predict --task regression --input examples/booking.json`.
- Document important errors and negative-prediction limitations; propose regression models and a training-fold tuning plan for Milestone 2.
- Own the regression fit/evaluate/save/load flow, the regression metric function, validation result export, and prediction/residual figure. Coordinate the regression experiment branch with Hala.
- Investigate residual patterns using training/validation data: systematic over/underprediction, larger errors by rate range, zero-rate cases, and the effect of rare/unseen booking categories. Describe error patterns without presenting them as causal findings.
- Verify saved-pipeline prediction consistency and task-relevant missing/unseen-input cases; supply regression examples and expected behavior to Hala for integration checks.
- Verify the regression assumptions, reference comparison, diagnostic interpretation, limitations, and next experiments in the result tables/figures and supporting notes.
- Completion evidence: reproduced fit/evaluation outputs, error analysis, inference checks, and links to actual changes or substantive review.

### MOHAMED HASSAN - complete classification workflow and imbalance analysis

- Review `src/classification/baseline.py`, classification metrics in `src/evaluation/metrics.py`, and classification plots in `src/experiments.py`.
- Reproduce the baseline workflow and verify the prior and logistic-regression results, positive prevalence, and confusion matrix.
- Explain accuracy, precision, recall, F1, ROC-AUC, average precision, and why the 0.5 threshold catches only 1 of 170 validation positives.
- Check saved classification inference with `python main.py predict --task classification --input examples/booking.json`.
- Document the poor default-threshold recall. Plan class weighting, calibration, and threshold evaluation using training-fold predictions for Milestone 2.
- Own the classification fit/evaluate/save/load flow, classification metric function, probability/decision outputs, validation result export, and confusion/ROC/precision-recall figures.
- Verify that the positive class, probability column, threshold, and reported precision/recall agree. Explain false negatives and false positives and compare ranking quality with the prior reference rather than relying on accuracy.
- Check saved-pipeline probability consistency and task-relevant input cases; supply classification examples and expected behavior to Hala for integration checks.
- Explain the narrowed risk set, rare-positive difficulty, poor baseline decisions, and the training-fold plan for weighting/calibration/threshold selection using the results and existing notes.
- Completion evidence: reproduced classification outputs, rare-positive diagnostic interpretation, inference checks, and links to actual changes or substantive review.

### AHMED MOKHTAR - complete clustering workflow and interpretation

- Review `src/clustering/baseline.py`, the clustering feature allowlist, and clustering analysis in `src/experiments.py`.
- Reproduce the baseline workflow and inspect `clustering_metrics.csv`, `cluster_profiles.csv`, and the cluster/PCA figure.
- Explain k=3 as an initial baseline, log transforms, scaling, one-hot categories, sampled silhouette, Davies-Bouldin, and the roughly 35% variance retained in the 2-D visualization.
- Interpret each cluster using booking attributes. Investigate sensitivity to rare-count weighting as planned work; keep outcome labels out of clustering decisions.
- Prepare the plan for optimized K-Means, hierarchical clustering, DBSCAN, and stability checks in Milestone 2.
- Own clustering-specific input preparation, fit/save flow, assignment export, unsupervised metrics, descriptive profile tables, and PCA display. Agree shared feature-policy boundaries with Omar.
- Check assignment coverage, profile counts, repeatability under the recorded seed, saved-pipeline assignment consistency, and the separation of clustering objects from supervised pipelines.
- Explain log/scaling/one-hot distance effects and how the projection's discarded variance limits interpretation. Do not describe the exploratory k=3 baseline as an optimized final choice.
- Verify the representation justification, metric limitations, segment descriptions, and remaining parameter/stability experiments using the generated results and existing notes.
- Completion evidence: reproduced assignments/profiles/metrics, verification of outcome isolation and artifacts, and links to actual changes or substantive review.

### HALA ELTAWEEL - CLI integration, system verification and reproducible delivery

- Review `main.py`, `src/inference.py`, `src/verification.py`, `README.md`, and `scripts/package_submission.py` against the measured outputs and current submission scope.
- Own common command routing in `main.py`, shared input handling in `src/inference.py`, and integrated verification in `src/verification.py`; coordinate task-specific prediction behavior with Ahmed Adel and Mohamed Hassan.
- Verify complete command flows: download/validate, analyze, baseline, verify, and both predictions. Check meaningful failures such as missing artifacts, malformed booking input, and incompatible source data, without adding final-test scoring.
- Follow the README setup/reproduction commands, check the six names and IDs, and verify that validation metrics are clearly separated from reserved test evaluation.
- Run `python main.py verify` and both example inference commands after baselines exist; check that the CLI and README agree on supported commands and expected output.
- Check source references, limitations, remaining-work plan, and the contribution evidence supplied by the other members.
- Own dependency/run instructions, output locations, command descriptions, interface consistency, and verification of the submission package. Confirm that the separate Milestone 1 report is absent.
- After final changes are committed, run `python scripts/package_submission.py --output ../Milestone1_Draft.zip`; check the recorded SHA and exclusion of downloaded data, environments, and fitted models.
- Completion evidence: integrated CLI and inference checks, reproduced environment/workflow evidence, checked run instructions, package verification, and links to actual changes or substantive review.

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
