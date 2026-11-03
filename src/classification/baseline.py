"""Untuned cancellation model and reference predictor."""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.preprocessing.pipelines import preprocessing


def models(seed):
    return {
        "training_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline([
            ("preprocessing", preprocessing("classification")),
            ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                         random_state=seed)),
        ]),
    }
