"""Export structural source summaries and training-only exploratory figures."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loading import CONFIG, CATEGORICAL
from src.features.policy import engineering

COLORS = ["#186B79", "#EF9A54", "#8670AD", "#5A8B65"]


def style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.titleweight": "bold", "savefig.dpi": 150})


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def analyze(frame, labels, output):
    style()
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    original = CONFIG["dataset"]["columns"]
    train = engineering(frame.loc[labels == "train"])
    schema = pd.DataFrame([
        {"column": col, "dtype": str(frame[col].dtype),
         "unknown_count": int(frame[col].isna().sum()),
         "not_applicable_count": int((frame[col] == "NOT_APPLICABLE").sum()) if col in CATEGORICAL else 0,
         "unique_nonmissing": int(frame[col].nunique())} for col in original])
    schema.to_csv(tables / "source_schema.csv", index=False)
    numeric = [col for col in original if col not in CATEGORICAL and col != "ReservationStatusDate"]
    train[numeric].describe(percentiles=[.01, .25, .5, .75, .99]).T.to_csv(tables / "training_numeric_summary.csv")
    outliers = []
    for col in numeric:
        values = train[col].dropna()
        q1, q3 = values.quantile([.25, .75])
        iqr = q3 - q1
        outliers.append({"column": col, "skewness": float(values.skew()),
                         "iqr_low": float(q1 - 1.5 * iqr), "iqr_high": float(q3 + 1.5 * iqr),
                         "outside_iqr_count": int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())})
    pd.DataFrame(outliers).to_csv(tables / "training_outliers.csv", index=False)
    categories = []
    for col in CATEGORICAL:
        for value, count in train[col].fillna("UNKNOWN").value_counts().items():
            categories.append({"column": col, "value": value, "count": int(count)})
    pd.DataFrame(categories).to_csv(tables / "training_category_counts.csv", index=False)
    guest_sum = frame[["Adults", "Children", "Babies"]].sum(axis=1, min_count=3)
    nights = frame.StaysInWeekendNights + frame.StaysInWeekNights
    summary = {
        "source_rows": len(frame), "source_columns": len(original),
        "arrival_first": str(frame.ArrivalDate.min().date()), "arrival_last": str(frame.ArrivalDate.max().date()),
        "exact_duplicate_rows_after_first": int(frame[original].duplicated().sum()),
        "unknown_counts": {col: int(frame[col].isna().sum()) for col in original if frame[col].isna().any()},
        "not_applicable_ids": {col: int((frame[col] == "NOT_APPLICABLE").sum()) for col in ["Agent", "Company"]},
        "structural_flags": {"zero_night_bookings": int((nights == 0).sum()),
                             "zero_guest_bookings": int((guest_sum == 0).sum()),
                             "adults_zero": int((frame.Adults == 0).sum()),
                             "negative_count_values": int((frame[[c for c in numeric if c not in ["ADR", "IsCanceled"]]] < 0).sum().sum())},
        "training_rows": len(train), "training_cancellation_rate": float(train.IsCanceled.mean()),
        "training_adr": {"mean": float(train.ADR.mean()), "median": float(train.ADR.median()),
                         "min": float(train.ADR.min()), "max": float(train.ADR.max()),
                         "zero_count": int((train.ADR == 0).sum()), "skewness": float(train.ADR.skew())},
        "duplicate_policy": "Retain repeated records: source lacks booking IDs; repeats may be real group bookings. Chronological cohorts preserve same-date predictor groups.",
        "eda_scope": "Source-wide schema, missingness and structural flags; target distributions and associations use training rows only.",
    }
    (output / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, col, title in zip(axes, ["ADR", "LeadTime", "TotalNights"], ["Recorded ADR", "Lead time (days)", "Total booked nights"]):
        cap = train[col].quantile(.99)
        values = train.loc[train[col] <= cap, col]
        ax.hist(values, bins=35, color=COLORS[0], alpha=.9)
        ax.set(title=title, ylabel="Training records", xlabel=f"Up to 99th percentile ({cap:.1f}); tail retained")
    fig.suptitle("Training distributions - display limits do not remove observations", y=1.04)
    fig.tight_layout()
    save(fig, figures / "training_distributions.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, col in zip(axes, ["MarketSegment", "Country", "ReservedRoomType"]):
        counts = train[col].fillna("UNKNOWN").value_counts().head(6).sort_values()
        ax.barh(counts.index, counts.values, color=COLORS[0])
        ax.set(title=col, xlabel="Training records (top six categories)")
    fig.tight_layout()
    save(fig, figures / "training_categories.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    unknown = schema.set_index("column")["unknown_count"]
    unknown = unknown[unknown > 0].sort_values()
    axes[0].barh(unknown.index, unknown.values, color=COLORS[0])
    axes[0].set(title="Unknown values: full-source structural check", xlabel="Records")
    counts = train.IsCanceled.value_counts().reindex([0, 1], fill_value=0)
    axes[1].bar(["Not canceled", "Late cancellation / no-show"], counts.values, color=COLORS[:2])
    axes[1].set(title="Training target after eligibility filtering", ylabel="Records")
    axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    save(fig, figures / "missingness_and_training_target.png")

    corr_cols = ["LeadTime", "TotalNights", "Adults", "Children", "Babies", "IsRepeatedGuest", "ADR", "IsCanceled"]
    corr = train[corr_cols].corr()
    corr.to_csv(tables / "training_correlations.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_cols)), corr_cols)
    ax.set_title("Training-only Pearson associations (descriptive)")
    for row in range(len(corr_cols)):
        for col in range(len(corr_cols)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iloc[row, col]) > .55 else "#222222")
    fig.colorbar(im, ax=ax, fraction=.04)
    save(fig, figures / "training_correlations.png")

    fig, ax = plt.subplots(figsize=(10, 3.1))
    cohorts = ["train", "validation", "test"]
    for i, name in enumerate(cohorts):
        series = frame.loc[labels == name].groupby(frame.ArrivalDate.dt.to_period("M")).size()
        ax.bar(series.index.astype(str), series.values, label=name, color=COLORS[i])
    ax.set(title="Supervised arrival cohorts - no test-target metrics", ylabel="Eligible records")
    ax.tick_params(axis="x", rotation=70, labelsize=7)
    ax.legend(frameon=False)
    save(fig, figures / "arrival_cohorts.png")
    return summary
