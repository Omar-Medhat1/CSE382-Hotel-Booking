"""Run reproducible Milestone 1 baselines without evaluating final test labels."""
import importlib.metadata
import json
import platform
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
                             davies_bouldin_score, silhouette_score)
from threadpoolctl import threadpool_limits

from src.analysis.eda import analyze, style, save, COLORS
from src.classification.baseline import models as classification_models
from src.clustering.baseline import model as cluster_model
from src.data.loading import ROOT, CONFIG, load_source
from src.data.splitting import partition
from src.evaluation.metrics import regression, classification
from src.features.policy import select, engineering
from src.regression.baseline import models as regression_models


def run():
    frame, validation = load_source()
    labels, manifest, split_info = partition(frame)
    output = ROOT / "results"
    tables = output / "tables"
    figures = output / "figures"
    artifacts = ROOT / "artifacts"
    for path in (tables, figures, artifacts):
        path.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(tables / "split_manifest.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    (output / "split_metadata.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    analysis = analyze(frame, labels, output)
    train = frame.loc[labels == "train"]
    valid = frame.loc[labels == "validation"]
    seed = CONFIG["seed"]
    rows = []
    details = {}
    fitted = {}
    print("Partitions:", split_info["partition_counts"], flush=True)
    # The final test partition is deliberately never passed to an estimator.
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for task, candidates, target, scorer in (
            ("regression", regression_models(seed), "ADR", regression),
            ("classification", classification_models(seed), "IsCanceled", classification),
        ):
            x_train, x_valid = select(train, task), select(valid, task)
            for name, estimator in candidates.items():
                start = time.perf_counter()
                estimator.fit(x_train, train[target])
                elapsed = time.perf_counter() - start
                prediction = (estimator.predict(x_valid) if task == "regression"
                              else estimator.predict_proba(x_valid)[:, 1])
                metrics = scorer(valid[target], prediction)
                matrix = metrics.pop("confusion_matrix", None)
                row = {"task": task, "model": name, "partition": "validation", "configuration": "untuned",
                       "training_rows": len(train), "validation_rows": len(valid),
                       "fit_seconds": elapsed, **metrics}
                rows.append(row)
                model_parameters = (estimator.named_steps["model"].get_params(deep=False)
                                    if hasattr(estimator, "named_steps") else estimator.get_params(deep=False))
                details[name] = {**row, "confusion_matrix": matrix, "parameters": model_parameters}
                print(name, metrics, flush=True)
                if name not in ("training_mean", "training_prior"):
                    fitted[task] = estimator
                    joblib.dump(estimator, artifacts / f"{task}_baseline.joblib")
                    pd.DataFrame({"source_row": valid.index[:10], "prediction": prediction[:10]}).to_csv(
                        tables / f"{task}_reload_reference.csv", index=False)
                    if task == "regression":
                        fig, ax = plt.subplots(figsize=(5.6, 4))
                        ax.scatter(valid[target], prediction, s=7, alpha=.25, color=COLORS[0], rasterized=True)
                        low = min(float(valid[target].min()), float(prediction.min()))
                        high = max(float(valid[target].max()), float(prediction.max()))
                        ax.plot([low, high], [low, high], "--", color=COLORS[1])
                        ax.set(xlabel="Observed validation ADR", ylabel="Predicted ADR", title="Untuned Ridge: validation predictions")
                        save(fig, figures / "regression_validation.png")
                    else:
                        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
                        ConfusionMatrixDisplay(np.array(matrix), display_labels=["Not canceled", "Late cancel"]).plot(ax=axes[0], colorbar=False, cmap="Blues")
                        RocCurveDisplay.from_predictions(valid[target], prediction, ax=axes[1], curve_kwargs={"color": COLORS[0]})
                        PrecisionRecallDisplay.from_predictions(valid[target], prediction, ax=axes[2], curve_kwargs={"color": COLORS[0]})
                        axes[2].axhline(valid[target].mean(), color=COLORS[1], linestyle="--", label="Validation prevalence")
                        axes[2].legend(fontsize=7)
                        for ax in axes:
                            ax.tick_params(labelsize=8)
                        axes[0].tick_params(axis="x", rotation=15)
                        fig.tight_layout()
                        save(fig, figures / "classification_validation.png")
        # Clustering is a separate full-source descriptive analysis. Its fitted
        # transforms and assignments are never used by the supervised workflows.
        x_cluster = select(frame, "clustering")
        clustering = cluster_model(seed)
        start = time.perf_counter()
        cluster_labels = clustering.fit_predict(x_cluster)
        cluster_seconds = time.perf_counter() - start
        transformed = clustering.named_steps["preprocessing"].transform(x_cluster).toarray()
        silhouette = silhouette_score(transformed, cluster_labels, sample_size=min(2000, len(frame)), random_state=seed)
        db = davies_bouldin_score(transformed, cluster_labels)
        cluster_metrics = {"model": "kmeans", "configuration": "untuned", "n_clusters": 3,
                           "rows": len(frame), "representation_dimensions": transformed.shape[1],
                           "silhouette_sample_size": min(2000, len(frame)), "silhouette": float(silhouette),
                           "davies_bouldin": float(db), "inertia": float(clustering.named_steps["model"].inertia_),
                           "fit_seconds": cluster_seconds,
                           "n_iter": int(clustering.named_steps["model"].n_iter_)}
        print("Clustering:", cluster_metrics, flush=True)
        joblib.dump(clustering, artifacts / "clustering_baseline.joblib")
        profile_frame = engineering(frame).assign(cluster=cluster_labels)
        profiles = profile_frame.groupby("cluster").agg(
            records=("LeadTime", "size"), median_lead_days=("LeadTime", "median"),
            median_nights=("TotalNights", "median"), mean_adults=("Adults", "mean"),
            mean_children=("Children", "mean"), mean_babies=("Babies", "mean"),
            median_weekend_share=("WeekendShare", "median"),
            top_market_segment=("MarketSegment", lambda x: x.mode().iloc[0]),
            top_customer_type=("CustomerType", lambda x: x.mode().iloc[0]),
            top_room_type=("ReservedRoomType", lambda x: x.mode().iloc[0]))
        profiles.to_csv(tables / "cluster_profiles.csv")
        pd.DataFrame({"source_row": frame.index, "cluster": cluster_labels}).to_csv(tables / "cluster_assignments.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        pca = PCA(n_components=2, random_state=seed)
        points = pca.fit_transform(transformed)
        joblib.dump(pca, artifacts / "cluster_visualization_pca.joblib")
        cluster_metrics["plot_pca_explained_variance"] = pca.explained_variance_ratio_.tolist()
        sample = np.random.default_rng(seed).choice(len(frame), min(8000, len(frame)), replace=False)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for number in range(3):
            ix = sample[cluster_labels[sample] == number]
            axes[0].scatter(points[ix, 0], points[ix, 1], s=5, alpha=.3, color=COLORS[number], label=f"Cluster {number}", rasterized=True)
        axes[0].set(xlabel="PC1", ylabel="PC2", title="Full-H2 clusters: 8,000-point display sample")
        axes[0].legend(markerscale=2, fontsize=8)
        axes[1].bar(profiles.index.astype(str), profiles.records, color=COLORS[:3])
        axes[1].set(xlabel="Cluster", ylabel="Records", title="Cluster sizes (all 79,330 rows)")
        fig.tight_layout()
        save(fig, figures / "clustering_baseline.png")
    pd.DataFrame(rows).to_csv(tables / "supervised_validation_metrics.csv", index=False)
    pd.DataFrame([cluster_metrics]).to_csv(tables / "clustering_metrics.csv", index=False)
    versions = {name: importlib.metadata.version(name) for name in
                ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl", "reportlab"]}
    run_info = {"python": platform.python_version(), "packages": versions, "seed": seed,
                "thread_limit": 2, "source": {**validation, "path": CONFIG["dataset"]["path"]}, "split": split_info, "eda": analysis,
                "models": details, "clustering": cluster_metrics,
                "validation_cancellation_rate": float(valid.IsCanceled.mean()),
                "final_test_evaluated": False,
                "limitations": ["Late cancellation/no-show risk set only, not booking-creation forecasting.",
                                "Source has no booking/customer IDs; independence cannot be fully established.",
                                "Arrival cohorts are an offline temporal evaluation, not rolling production simulation.",
                                "Full-H2 unsupervised outputs never feed supervised models or model selection."]}
    (output / "run_summary.json").write_text(json.dumps(run_info, indent=2, default=str), encoding="utf-8")
    return run_info
