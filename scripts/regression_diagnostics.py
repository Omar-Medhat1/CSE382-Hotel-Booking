"""Check Ridge prediction errors using the saved model and validation data only.

Run: python -m scripts.regression_diagnostics
No retraining and no final test evaluation.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loading import ROOT, load_source
from src.data.splitting import partition
from src.evaluation.metrics import regression
from src.features.policy import CATEGORICAL, select


ADR_BINS = [-np.inf, 0, 50, 100, 150, 200, np.inf]
ADR_LABELS = ["ADR = 0", "0 < ADR <= 50", "50 < ADR <= 100",
              "100 < ADR <= 150", "150 < ADR <= 200", "ADR > 200"]


def category_familiarity(train, valid, rare_below=10):
    """Group categories based on training counts, not validation counts.

    Unseen takes priority if a row has both rare and unseen values.
    """
    if rare_below < 2:
        raise ValueError("rare_below must be at least 2")
    unseen = np.zeros(len(valid), dtype=bool)
    rare = np.zeros(len(valid), dtype=bool)
    for column in CATEGORICAL:
        counts = train[column].dropna().value_counts()
        frequencies = valid[column].map(counts).to_numpy(dtype=float, na_value=np.nan)
        nonmissing = valid[column].notna().to_numpy()
        unseen |= nonmissing & ~np.isfinite(frequencies)
        rare |= nonmissing & np.isfinite(frequencies) & (frequencies < rare_below)
    return pd.Series(np.where(unseen, "at_least_one_unseen",
                              np.where(rare, "at_least_one_rare", "no_rare_or_unseen")),
                     index=valid.index, name="category_familiarity")


def summarize_residuals(actual, predicted, categories):
    """Calculate error statistics for each validation group."""
    y = np.asarray(actual, dtype=float).reshape(-1)
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    if len(y) != len(pred) or len(y) != len(categories) or not len(y):
        raise ValueError("Expected equal nonempty validation arrays")
    if not np.isfinite(y).all() or not np.isfinite(pred).all() or (y < 0).any():
        raise ValueError("ADR values must be finite, and observed ADR nonnegative")

    residual = y - pred  # Positive means underprediction, negative means overprediction.
    bands = pd.cut(y, bins=ADR_BINS, labels=ADR_LABELS)
    groups = [("overall", "all", np.ones(len(y), dtype=bool))]
    groups += [("observed_adr_band", label, np.asarray(bands == label)) for label in ADR_LABELS]
    familiarity = np.asarray(categories)
    groups += [("training_category_familiarity", label, familiarity == label)
               for label in ("no_rare_or_unseen", "at_least_one_rare", "at_least_one_unseen")]

    records = []
    for group, segment, selector in groups:
        count = int(selector.sum())
        if count == 0:
            continue
        diff = residual[selector]
        records.append({"group": group, "segment": segment, "count": count,
                        "mean_observed_adr": float(y[selector].mean()),
                        "mean_predicted_adr": float(pred[selector].mean()),
                        "mean_residual_actual_minus_predicted": float(diff.mean()),
                        "mae": float(np.abs(diff).mean()),
                        "rmse": float(np.sqrt(np.mean(diff ** 2))),
                        "negative_predictions": int(np.sum(pred[selector] < 0))})
    return pd.DataFrame.from_records(records)


def make_figure(actual, predicted, summary, destination):
    """Plot residuals and MAE for each ADR range."""
    y = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].scatter(pred, y - pred, alpha=.18, s=8, rasterized=True)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Predicted ADR", ylabel="Residual = actual - predicted",
                title="Ridge validation residuals")

    b = summary.loc[summary.group == "observed_adr_band"]
    axes[1].bar(range(len(b)), b.mae)
    axes[1].set_xticks(range(len(b)), b.segment, rotation=32, ha="right")
    axes[1].set(ylabel="MAE (ADR units)", title="Validation error by observed ADR")
    for i, (_, row) in enumerate(b.iterrows()):
        axes[1].text(i, row.mae, f"n={int(row['count'])}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def run():
    frame, _ = load_source()  # Check the source file before using it.
    labels, _, _ = partition(frame)
    train = frame.loc[labels == "train"]
    valid = frame.loc[labels == "validation"]
    artifact = ROOT / "artifacts" / "regression_baseline.joblib"
    if not artifact.is_file():
        raise FileNotFoundError(f"Saved Ridge pipeline missing: {artifact}. Run main.py baseline first.")
    model = joblib.load(artifact)  # Use the locally saved model.
    predicted = model.predict(select(valid, "regression"))
    metric = regression(valid.ADR, predicted)

    # Make sure the saved model matches the current baseline results.
    metric_file = ROOT / "results" / "tables" / "supervised_validation_metrics.csv"
    reference = pd.read_csv(metric_file)
    row = reference.loc[(reference.task == "regression") & (reference.model == "ridge") &
                        (reference.partition == "validation")]
    if len(row) != 1:
        raise ValueError("Could not find exactly one Ridge validation reference in metrics table")
    for name in ("mae", "mse", "rmse", "r2"):
        if not np.isclose(metric[name], float(row.iloc[0][name]), rtol=1e-7, atol=1e-7):
            raise ValueError(f"Saved pipeline disagrees with recorded validation {name}; check your baseline outputs")

    familiarity = category_familiarity(train, valid)
    summary = summarize_residuals(valid.ADR, predicted, familiarity)
    table_path = ROOT / "results" / "tables" / "regression_validation_residual_summary.csv"
    figure_path = ROOT / "results" / "figures" / "regression_validation_residuals.png"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False, float_format="%.8f")
    make_figure(valid.ADR, predicted, summary, figure_path)
    report = {"scope": "validation only; no final test labels used",
              "validation_rows": len(valid), "ridge_validation_metrics": metric,
              "mean_residual_actual_minus_predicted": float(np.mean(valid.ADR.to_numpy() - predicted)),
              "negative_predictions": int(np.sum(predicted < 0)),
              "rare_threshold": "fewer than 10 training examples per nonmissing category",
              "saved_pipeline_retrained": False,
              "residual_table": str(table_path.relative_to(ROOT)),
              "residual_figure": str(figure_path.relative_to(ROOT))}
    print(json.dumps(report, indent=2))
    return summary


if __name__ == "__main__":
    run()
