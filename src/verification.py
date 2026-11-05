"""Check experimental integrity and saved inference artifacts."""

import json
import joblib
import numpy as np
import pandas as pd

from src.data.loading import ROOT, load_source
from src.data.splitting import partition
from src.features.policy import select, NUMERIC


def require(condition, message):
    """Raise an error when a verification check fails."""
    if not condition:
        raise ValueError(message)


def verify():
    frame, source = load_source()
    labels, manifest, metadata = partition(frame)

    recorded = pd.read_csv(
        ROOT / "results/tables/split_manifest.csv.gz",
        dtype={"predictor_group": str}
    )

    # Verify data splitting.
    require(
        manifest.partition.tolist() == recorded.partition.tolist(),
        "Split assignments changed."
    )

    summary = json.loads(
        (ROOT / "results/run_summary.json").read_text()
    )

    # Verify source dataset integrity.
    require(
        summary["source"]["sha256"] == source["sha256"],
        "Source checksum changed."
    )

    # Ensure final test evaluation remains reserved.
    require(
        summary["final_test_evaluated"] is False,
        "Final test must remain reserved."
    )

    train = frame.loc[labels == "train"]

    # ==========================================================
    # Regression and classification verification
    # ==========================================================

    for task in ("regression", "classification"):

        model = joblib.load(
            ROOT / "artifacts" / f"{task}_baseline.joblib"
        )

        reference = pd.read_csv(
            ROOT / "results/tables" / f"{task}_reload_reference.csv"
        )

        inputs = select(
            frame.loc[reference.source_row],
            task
        )

        if task == "regression":
            actual = model.predict(inputs)
        else:
            actual = model.predict_proba(inputs)[:, 1]

        # Verify saved model predictions.
        np.testing.assert_allclose(
            actual,
            reference.prediction,
            rtol=1e-10,
            atol=1e-10
        )

        # Verify training-only imputation.
        imputer = (
            model.named_steps["preprocessing"]
            .named_transformers_["numeric"]
            .named_steps["imputer"]
        )

        np.testing.assert_allclose(
            imputer.statistics_,
            select(train, task)[NUMERIC].median().values
        )

        # Verify unseen-category handling.
        encoder = (
            model.named_steps["preprocessing"]
            .named_transformers_["categorical"]
            .named_steps["encoder"]
        )

        require(
            encoder.handle_unknown == "ignore",
            "Encoder must ignore unseen categories."
        )

    # Verify chronological label cutoffs.
    require(
        (train.StatusDate < pd.Timestamp("2016-12-31")).all(),
        "Training labels violate cutoff."
    )

    require(
        (
            frame.loc[labels == "validation", "StatusDate"]
            < pd.Timestamp("2017-04-30")
        ).all(),
        "Validation labels violate cutoff."
    )

    # ==========================================================
    # Clustering verification - Ahmed Mokhtar
    # ==========================================================

    # Reload the saved clustering pipeline.
    clustering_model = joblib.load(
        ROOT / "artifacts/clustering_baseline.joblib"
    )

    # Load clustering assignments and profiles.
    assignments = pd.read_csv(
        ROOT / "results/tables/cluster_assignments.csv.gz"
    )

    profiles = pd.read_csv(
        ROOT / "results/tables/cluster_profiles.csv"
    )

    # Verify that all source rows have exactly one assignment.
    require(
        len(assignments) == len(frame) == 79330,
        "Clustering row coverage mismatch."
    )

    require(
        assignments.source_row.is_unique,
        "Duplicate source rows in clustering assignments."
    )

    np.testing.assert_array_equal(
        assignments.source_row.to_numpy(),
        frame.index.to_numpy(),
        err_msg="Clustering source-row order mismatch."
    )

    # Verify that the reloaded model reproduces assignments.
    clustering_inputs = select(frame, "clustering")

    predicted_clusters = clustering_model.predict(
        clustering_inputs
    )

    np.testing.assert_array_equal(
        predicted_clusters,
        assignments.cluster.to_numpy(),
        err_msg="Reloaded clustering predictions differ."
    )

    # Verify cluster profile counts.
    actual_counts = (
        assignments.cluster.value_counts().sort_index()
    )

    expected_counts = (
        profiles.set_index("cluster")["records"].sort_index()
    )

    pd.testing.assert_series_equal(
        actual_counts,
        expected_counts,
        check_names=False
    )

    # ==========================================================
    # Verification summary
    # ==========================================================

    result = {
        "source_checksum": source["sha256"],
        "split_reproduction": "passed",
        "predictor_group_overlap": metadata["predictor_group_overlap"],
        "train_only_imputation": "passed",
        "artifact_reload_predictions": "passed",
        "clustering_reload_and_coverage": "passed",
        "final_test_evaluated": False
    }

    (ROOT / "results/verification.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    return result