"""Fixed arrival-cohort holdouts with label-availability purging."""
import numpy as np
import pandas as pd

from src.features.policy import predictor_groups


def partition(frame):
    cutoff = frame.ArrivalDate - pd.Timedelta(days=1)
    # At the end of the snapshot day the reservation must exist and be active.
    # Events dated on the snapshot day are excluded because times are unavailable.
    eligible = (frame.LeadTime >= 1) & (frame.StatusDate > cutoff)
    val_start = pd.Timestamp("2017-01-01")
    test_start = pd.Timestamp("2017-05-01")
    labels = pd.Series("ineligible_at_snapshot", index=frame.index, dtype=object)
    labels.loc[eligible & (frame.ArrivalDate < val_start)] = "train"
    labels.loc[eligible & (frame.ArrivalDate >= val_start) & (frame.ArrivalDate < test_start)] = "validation"
    labels.loc[eligible & (frame.ArrivalDate >= test_start)] = "test"
    labels.loc[(labels == "train") & (frame.StatusDate >= val_start - pd.Timedelta(days=1))] = "purged_train_late_label"
    labels.loc[(labels == "validation") & (frame.StatusDate >= test_start - pd.Timedelta(days=1))] = "purged_validation_late_label"
    groups = predictor_groups(frame)
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = set(groups[labels == left]) & set(groups[labels == right])
        overlaps[f"{left}_{right}"] = len(overlap)
        if overlap:
            raise ValueError("Identical predictor groups crossed cohort boundaries.")
    for name in ("train", "validation", "test"):
        if not (labels == name).any():
            raise ValueError(f"Empty {name} partition.")
    # Inspect target balance only in the development partitions.
    for name in ("train", "validation"):
        if frame.loc[labels == name, "IsCanceled"].nunique() != 2:
            raise ValueError(f"{name} must contain both classification outcomes.")
    manifest = pd.DataFrame({"source_row": frame.index, "partition": labels,
                             "predictor_group": groups, "arrival_date": frame.ArrivalDate.dt.strftime("%Y-%m-%d")})
    metadata = {
        "protocol": "End-of-day before arrival; lead time >=1; final status strictly after snapshot day.",
        "train_arrival_before": "2017-01-01", "validation_arrival_before": "2017-05-01",
        "training_labels_before": "2016-12-31", "validation_labels_before": "2017-04-30",
        "partition_counts": {str(k): int(v) for k, v in labels.value_counts().items()},
        "predictor_group_overlap": overlaps,
        "final_test_evaluated": False,
    }
    return labels, manifest, metadata
