import numpy as np
from sklearn.cluster import KMeans


def fit(X: np.ndarray, k: int, seed=None, max_iter: int = 100, eps: float = 1e-4):
    """Fit k-means using scikit-learn's defaults (k-means++ init, best-of-n runs)

    Returns (centroids, labels, n_iter) (it's the same interface as kmeans.fit(), so it matches compressor.py)
    """
    km = KMeans(n_clusters=k, max_iter=max_iter, tol=eps, random_state=seed)
    km.fit(X)
    return km.cluster_centers_, km.labels_, km.n_iter_
