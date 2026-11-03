"""Independent full-H2 segmentation without target fields."""
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from src.preprocessing.pipelines import preprocessing


def model(seed):
    # k=3 is a fixed exploratory baseline, not the final selected cluster count.
    return Pipeline([("preprocessing", preprocessing("clustering")),
                     ("model", KMeans(n_clusters=3, n_init=10, random_state=seed))])
