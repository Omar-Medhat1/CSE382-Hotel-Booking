# CSE382 - City Hotel Booking Demand

Repository: https://github.com/Omar-Medhat1/CSE382-Hotel-Booking

## Current status

Initial repository setup for Milestone 1. The source-data validator, command-line entry point, configuration, and project structure are available. Data analysis, preprocessing, baseline experiments, and the progress report remain to be completed. No experimental results are claimed in this version.

The course brief gives the Milestone 1 deadline as 5 November 2026.

## Environment setup

Use Python 3.11 or newer. From the project directory on Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py status
```

The initial setup uses only the Python standard library. ML dependencies will be added and their versions recorded when the analysis and experiments are implemented and verified.

## Data

Use H2.csv only (City Hotel). Do not merge H1 and H2.

Download the CSV from the source mirror documented in the course brief:

https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv

Save it as `data/raw/H2.csv`, then run:

```powershell
.\.venv\Scripts\python.exe main.py validate-data
```

The validator checks the 31 expected source columns, record width, and 79,330 expected records. This structural check does not replace the later EDA, cleaning, or leakage audit. Preserve original source labels.

Dataset reference: Nuno Antonio, Ana de Almeida, and Luis Nunes, "Hotel booking demand datasets," Data in Brief 22 (2019), 41-49. https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/

## Layout

```text
main.py                 Command-line entry point
config.json             Data source, schema, and experiment seed
src/data/               Source-data loading and validation
src/preprocessing/      Reusable preprocessing (pending)
src/features/           Task-specific feature selection (pending)
src/regression/         ADR baseline (pending)
src/classification/     IsCanceled baseline (pending)
src/clustering/         Unsupervised baseline (pending)
src/evaluation/         Metrics and result tables (pending)
data/raw/               Downloaded H2.csv (excluded from Git)
artifacts/              Generated fitted objects (excluded from Git)
results/                Figures and measured results
reports/                Milestone progress reports
docs/                   Team details and implementation plan
```

## Milestone 1 work plan

1. Verify H2.csv and inspect types, missingness, duplicates, unusual values, category cardinality, distributions, and outliers.
2. Define the prediction or analysis point and feature availability separately for ADR regression, cancellation classification, and clustering.
3. Document and implement reproducible training/validation/test partitions. Keep final supervised test data untouched during Milestone 1 model selection.
4. Implement preprocessing pipelines with all learned supervised transformations fitted only on training data.
5. Run at least one untuned regression model, one classifier, and one clustering algorithm; report preliminary validation or unsupervised metrics and interpretations.
6. Complete the preliminary leakage audit, contribution record, progress report, and Milestone 2 plan.

## Team and contributions

Team details are pending; see `docs/team.md`. Record actual work in `docs/contributions.md`. Each member must contribute through their own GitHub account. Do not attribute unperformed work to a member.

## Submission

The final Milestone 1 ZIP must contain source code, configuration, README, dependencies, the completed progress report, and measured result tables/figures. Exclude trained models and fitted pipelines. The repository link is above. Record the submitted commit SHA when the submission is finalized, using `git rev-parse HEAD` after committing the submission contents.

Submitted commit SHA: PENDING - this is an initial setup, not the milestone submission.
