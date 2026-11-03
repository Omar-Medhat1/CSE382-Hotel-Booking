"""Explicit feature allowlists and stateless feature engineering."""
import numpy as np
import pandas as pd

NUMERIC = ["LeadTime", "ArrivalDateYear", "ArrivalDateDayOfMonth",
           "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children",
           "Babies", "IsRepeatedGuest", "TotalNights", "WeekendShare",
           "ArrivalMonthSin", "ArrivalMonthCos"]
CATEGORICAL = ["Meal", "Country", "MarketSegment", "DistributionChannel",
               "ReservedRoomType", "Agent", "Company", "CustomerType"]
GROUP_COLUMNS = ["LeadTime", "ArrivalDateYear", "ArrivalDateMonth",
                 "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights",
                 "Adults", "Children", "Babies", "IsRepeatedGuest", *CATEGORICAL]
CLUSTER_NUMERIC = ["LogLeadTime", "LogTotalNights", "Adults", "Children", "Babies", "WeekendShare"]
CLUSTER_CATEGORICAL = ["MarketSegment", "CustomerType", "ReservedRoomType"]
FORBIDDEN = {"ADR", "IsCanceled", "ReservationStatus", "ReservationStatusDate",
             "AssignedRoomType", "BookingChanges", "DepositType", "DaysInWaitingList",
             "RequiredCarParkingSpaces", "TotalOfSpecialRequests",
             "PreviousCancellations", "PreviousBookingsNotCanceled", "StatusDate"}


def engineering(frame):
    """No fitted statistics; safe to apply independently to any partition."""
    result = frame.copy()
    result["TotalNights"] = result.StaysInWeekendNights + result.StaysInWeekNights
    result["WeekendShare"] = result.StaysInWeekendNights / result.TotalNights.replace(0, np.nan)
    # A zero-night booking has zero booked weekend nights; define its share as 0.
    result["WeekendShare"] = result.WeekendShare.fillna(0.0)
    month = result.ArrivalDate.dt.month
    result["ArrivalMonthSin"] = np.sin(2 * np.pi * month / 12)
    result["ArrivalMonthCos"] = np.cos(2 * np.pi * month / 12)
    result["LogLeadTime"] = np.log1p(result.LeadTime)
    result["LogTotalNights"] = np.log1p(result.TotalNights)
    return result


def select(frame, task):
    if task in ("regression", "classification"):
        columns = NUMERIC + CATEGORICAL
    elif task == "clustering":
        columns = CLUSTER_NUMERIC + CLUSTER_CATEGORICAL
    else:
        raise ValueError(f"Unknown task: {task}")
    assert not (set(columns) & FORBIDDEN)
    # Object arrays with ordinary np.nan keep imputation behavior stable.
    result = engineering(frame)[columns].copy()
    for col in set(columns) & set(CATEGORICAL + CLUSTER_CATEGORICAL):
        result[col] = result[col].astype(object).where(result[col].notna(), np.nan)
    return result


def predictor_groups(frame):
    """Group identical raw allowlisted predictors without using target values."""
    return pd.util.hash_pandas_object(frame[GROUP_COLUMNS], index=False).astype(str)
