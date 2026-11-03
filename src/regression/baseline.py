"""Untuned ADR model and reference predictor."""
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from src.preprocessing.pipelines import preprocessing


def models(seed):
    return {
        "training_mean": DummyRegressor(strategy="mean"),
        "ridge": Pipeline([("preprocessing", preprocessing("regression")),
                            ("model", Ridge(alpha=1.0, solver="lsqr"))]),
    }
