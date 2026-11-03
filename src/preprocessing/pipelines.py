"""Train-only imputers, categorical encoders, and scaling."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.policy import NUMERIC, CATEGORICAL, CLUSTER_NUMERIC, CLUSTER_CATEGORICAL


def preprocessing(task):
    numeric, categorical = ((CLUSTER_NUMERIC, CLUSTER_CATEGORICAL) if task == "clustering"
                            else (NUMERIC, CATEGORICAL))
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)),
        ]), categorical),
    ], sparse_threshold=1.0)
