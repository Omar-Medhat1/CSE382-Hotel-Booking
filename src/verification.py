"""Check experimental integrity and saved inference artifacts."""
import json
import joblib
import numpy as np
import pandas as pd
from src.data.loading import ROOT, load_source
from src.data.splitting import partition
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
        actual = model.predict(inputs) if task == "regression" else model.predict_proba(inputs)[:, 1]
        np.testing.assert_allclose(actual, reference.prediction, rtol=1e-10, atol=1e-10)
        imputer = model.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["imputer"]
        np.testing.assert_allclose(imputer.statistics_, select(train, task)[NUMERIC].median().values)
        encoder = model.named_steps["preprocessing"].named_transformers_["categorical"].named_steps["encoder"]
        assert encoder.handle_unknown == "ignore"
    assert (train.StatusDate < pd.Timestamp("2016-12-31")).all()
    assert (frame.loc[labels == "validation", "StatusDate"] < pd.Timestamp("2017-04-30")).all()
    result = {"source_checksum": source["sha256"], "split_reproduction": "passed",
              "predictor_group_overlap": metadata["predictor_group_overlap"],
              "train_only_imputation": "passed", "artifact_reload_predictions": "passed",
              "final_test_evaluated": False}
    (ROOT / "results/verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
