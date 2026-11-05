"""Check experimental integrity and saved inference artifacts."""
import json
import joblib
import numpy as np
import pandas as pd
from src.data.loading import ROOT, load_source
from src.data.splitting import partition
from src.evaluation.metrics import (classification_decisions,
                                    classification_probability_summary)
from src.features.policy import select, NUMERIC


def verify():
    frame, source = load_source()
    labels, manifest, metadata = partition(frame)
    recorded = pd.read_csv(ROOT / "results/tables/split_manifest.csv.gz", dtype={"predictor_group": str})
    assert manifest.partition.tolist() == recorded.partition.tolist(), "Split assignments changed."
    summary = json.loads((ROOT / "results/run_summary.json").read_text())
    assert summary["source"]["sha256"] == source["sha256"], "Source checksum changed."
    assert summary["final_test_evaluated"] is False
    train = frame.loc[labels == "train"]
    for task in ("regression", "classification"):
        model = joblib.load(ROOT / "artifacts" / f"{task}_baseline.joblib")
        reference = pd.read_csv(ROOT / "results/tables" / f"{task}_reload_reference.csv")
        inputs = select(frame.loc[reference.source_row], task)
        if task == "regression":
            actual = model.predict(inputs)
        else:
            np.testing.assert_array_equal(model.classes_, [0, 1])
            probability_matrix = model.predict_proba(inputs)
            np.testing.assert_allclose(probability_matrix.sum(axis=1), 1.0)
            actual = probability_matrix[:, 1]
            np.testing.assert_array_equal(model.predict(inputs), classification_decisions(actual))
        np.testing.assert_allclose(actual, reference.prediction, rtol=1e-10, atol=1e-10)
        imputer = model.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["imputer"]
        np.testing.assert_allclose(imputer.statistics_, select(train, task)[NUMERIC].median().values)
        encoder = model.named_steps["preprocessing"].named_transformers_["categorical"].named_steps["encoder"]
        assert encoder.handle_unknown == "ignore"
        if task == "classification":
            valid = frame.loc[labels == "validation"]
            validation_probability = model.predict_proba(select(valid, task))[:, 1]
            expected_summary = classification_probability_summary(valid.IsCanceled, validation_probability)
            recorded_summary = pd.read_csv(
                ROOT / "results/tables/classification_validation_probability_summary.csv")
            pd.testing.assert_frame_equal(expected_summary, recorded_summary, check_exact=False,
                                          rtol=1e-12, atol=1e-12)
    assert (train.StatusDate < pd.Timestamp("2016-12-31")).all()
    assert (frame.loc[labels == "validation", "StatusDate"] < pd.Timestamp("2017-04-30")).all()
    result = {"source_checksum": source["sha256"], "split_reproduction": "passed",
              "predictor_group_overlap": metadata["predictor_group_overlap"],
              "train_only_imputation": "passed", "artifact_reload_predictions": "passed",
              "classification_positive_class": "class 1 at probability column 1",
              "classification_probability_decision_consistency": "passed",
              "classification_probability_summary": "passed",
              "final_test_evaluated": False}
    (ROOT / "results/verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
