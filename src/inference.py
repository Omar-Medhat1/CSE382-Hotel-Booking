"""Load an already fitted baseline; never retrain during prediction."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.data.loading import ROOT, MONTHS
from src.features.policy import GROUP_COLUMNS, CATEGORICAL, select


def predict(task, path):
    booking = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(booking, dict):
        raise ValueError("Booking input must be a JSON object.")
    expected = set(GROUP_COLUMNS)
    if set(booking) != expected:
        raise ValueError(f"Missing keys: {sorted(expected-set(booking))}; unexpected keys: {sorted(set(booking)-expected)}")
    frame = pd.DataFrame([booking])
    for col in expected:
        if col in CATEGORICAL or col == "ArrivalDateMonth":
            value = frame.at[0, col]
            value = str(value).strip() if value is not None else ""
            if col in ("Agent", "Company"):
                value = "NOT_APPLICABLE" if value in ("NULL", "", "NOT_APPLICABLE") else value
            elif value in ("NA", "NULL", ""):
                value = np.nan
            frame[col] = pd.Series([value], dtype=object)
        else:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame["ArrivalDate"] = pd.to_datetime(dict(year=frame.ArrivalDateYear,
                                             month=frame.ArrivalDateMonth.map(MONTHS),
                                             day=frame.ArrivalDateDayOfMonth), errors="raise")
    if frame.LeadTime.iloc[0] < 1:
        raise ValueError("Baseline is defined for bookings already active the day before arrival (LeadTime >=1).")
    artifact = ROOT / "artifacts" / f"{task}_baseline.joblib"
    model = joblib.load(artifact)
    x = select(frame, task)
    if task == "regression":
        return {"task": task, "model": "untuned Ridge", "predicted_adr": float(model.predict(x)[0]),
                "scope": "Recorded ADR estimate for an active booking at the prior-day snapshot."}
    probability = float(model.predict_proba(x)[0, 1])
    return {"task": task, "model": "untuned logistic regression", "cancellation_probability": probability,
            "prediction_at_threshold_0_5": int(probability >= .5),
            "scope": "Late cancellation/no-show among bookings active at the end of the day before arrival."}
