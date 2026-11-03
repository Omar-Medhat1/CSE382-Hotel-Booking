"""Download, normalize, and validate the original City Hotel source."""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .validation import validate_source

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
CATEGORICAL = ["ArrivalDateMonth", "Meal", "Country", "MarketSegment",
               "DistributionChannel", "ReservedRoomType", "AssignedRoomType",
               "DepositType", "Agent", "Company", "CustomerType", "ReservationStatus"]
MONTHS = {name: i for i, name in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], 1)}


def download():
    path = ROOT / CONFIG["dataset"]["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return validate_source(path, CONFIG["dataset"])
    temporary = path.with_suffix(".csv.part")
    try:
        with urllib.request.urlopen(CONFIG["dataset"]["url"], timeout=90) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        validate_source(temporary, CONFIG["dataset"])
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return validate_source(path, CONFIG["dataset"])


def load_source():
    path = ROOT / CONFIG["dataset"]["path"]
    validation = validate_source(path, CONFIG["dataset"])
    frame = pd.read_csv(path, dtype="str", keep_default_na=False)
    for col in CATEGORICAL:
        frame[col] = frame[col].str.strip()
        if col in ("Agent", "Company"):
            # The paper defines NULL IDs as not applicable, not unknown.
            frame[col] = frame[col].replace({"NULL": "NOT_APPLICABLE", "": "NOT_APPLICABLE"})
        else:
            frame[col] = frame[col].replace({"NULL": np.nan, "NA": np.nan, "": np.nan})
    for col in CONFIG["dataset"]["columns"]:
        if col not in CATEGORICAL and col != "ReservationStatusDate":
            normalized = frame[col].str.strip().replace({"NA": np.nan, "NULL": np.nan, "": np.nan})
            frame[col] = pd.to_numeric(normalized, errors="raise")
    frame["ArrivalDate"] = pd.to_datetime(dict(
        year=frame.ArrivalDateYear, month=frame.ArrivalDateMonth.map(MONTHS),
        day=frame.ArrivalDateDayOfMonth), errors="raise")
    frame["StatusDate"] = pd.to_datetime(frame.ReservationStatusDate, errors="raise")
    if not frame.IsCanceled.isin([0, 1]).all():
        raise ValueError("IsCanceled must be binary.")
    if not np.isfinite(frame.ADR).all() or (frame.ADR < 0).any():
        raise ValueError("Unexpected ADR domain: inspect source before modeling.")
    if not frame.ReservationStatus.isin(["Check-Out", "Canceled", "No-Show"]).all():
        raise ValueError("Unknown reservation status.")
    expected_labels = frame.ReservationStatus.isin(["Canceled", "No-Show"]).astype(int)
    if not (frame.IsCanceled == expected_labels).all():
        raise ValueError("Outcome status and cancellation target disagree.")
    return frame, validation
