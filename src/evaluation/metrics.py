"""Validation-only supervised metrics and unsupervised evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)


def regression(y, prediction):
    mse = mean_squared_error(y, prediction)
    return {"mae": float(mean_absolute_error(y, prediction)), "mse": float(mse),
            "rmse": float(mse ** 0.5), "r2": float(r2_score(y, prediction))}


def classification_decisions(probability, threshold=0.5):
    """Convert positive-class probabilities to binary decisions consistently."""
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 1 or not np.isfinite(probability).all():
        raise ValueError("Classification probabilities must be a finite one-dimensional array.")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Classification probabilities must be between 0 and 1.")
    if not 0 <= threshold <= 1:
        raise ValueError("Classification threshold must be between 0 and 1.")
    return (probability >= threshold).astype(int)


def classification(y, probability):
    prediction = classification_decisions(probability)
    return {"accuracy": float(accuracy_score(y, prediction)),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "f1": float(f1_score(y, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist()}


def classification_probability_summary(y, probability):
    """Summarize validation probabilities separately for each true class."""
    probability = np.asarray(probability, dtype=float)
    decisions = classification_decisions(probability)
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(probability):
        raise ValueError("Labels and probabilities must be aligned one-dimensional arrays.")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Classification labels must be binary values 0 and 1.")
    frame = pd.DataFrame({"true_class": labels.astype(int), "probability": probability,
                          "decision_at_0_5": decisions})
    rows = []
    for label, group in frame.groupby("true_class", sort=True):
        values = group.probability
        rows.append({"true_class": int(label), "class_name": "late_cancel" if label else "not_canceled",
                     "rows": int(len(group)), "mean_probability": float(values.mean()),
                     "std_probability": float(values.std(ddof=0)), "min_probability": float(values.min()),
                     "p25_probability": float(values.quantile(.25)), "median_probability": float(values.median()),
                     "p75_probability": float(values.quantile(.75)), "p90_probability": float(values.quantile(.90)),
                     "p95_probability": float(values.quantile(.95)), "max_probability": float(values.max()),
                     "decisions_at_0_5": int(group.decision_at_0_5.sum()),
                     "decision_rate_at_0_5": float(group.decision_at_0_5.mean())})
    return pd.DataFrame(rows)
