from __future__ import annotations

import numpy as np
from tqdm import tqdm

KMEANS_DEFAULTS = {"seed": None, "max_iter": 100, "eps": 1e-4, "batch_size": 100_000}


def _initialize_centroids(X: np.ndarray, k: int, seed: int | None = None) -> np.ndarray:
    """Initialize an array with k-random starting centroids"""
    rng = np.random.default_rng(seed)
    centroids = rng.choice(X, k, replace=False)
    return centroids


def _assign_clusters(X: np.ndarray, centroids: np.ndarray, batch_size: int = 100_000):
    """Assign labels to the nearest centroids creating clusters, returns (squared distance, id of nearest centroid)"""
    n = len(X)
    distances = np.empty(n)
    labels = np.empty(n, dtype=int)

    for i in range(
        0, n, batch_size
    ):  # introduce batching to solve the ArrayMemoryError
        end = min(i + batch_size, n)
        batch = X[i:end]

        batch_dist = (
            (batch**2).sum(axis=1).reshape(-1, 1)
            + (centroids**2).sum(axis=1).reshape(1, -1)
            - 2 * np.dot(batch, centroids.T)
        )

        batch_labels = np.argmin(batch_dist, axis=1)

        distances[i:end] = np.min(batch_dist, axis=1)
        labels[i:end] = batch_labels
    return distances, labels


def _update_centroids(
    X: np.ndarray, labels: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Update centroids by finding their means in (r,g,b) space. Returns np.ndarray with new positions of centroids"""
    new_centroids = np.empty_like(centroids, dtype=float)

    for cluster_id in range(len(centroids)):
        mask = labels == cluster_id
        # if labels is the same as cluster_id, returns list of [True, False, True, ... etc.]
        if np.any(mask):
            new_centroids[cluster_id] = X[mask].mean(axis=0)
        else:
            new_centroids[cluster_id] = centroids[cluster_id]
    return new_centroids


def fit(
    X: np.array,
    k: int,
    seed: int | None = KMEANS_DEFAULTS["seed"],
    max_iter: int = KMEANS_DEFAULTS["max_iter"],
    eps: float = KMEANS_DEFAULTS["eps"],
    batch_size: int = KMEANS_DEFAULTS["batch_size"],
):
    """Run k-means clustering algorithm. Returns (centroids, labels, n_iter)"""
    X = X.astype(np.float64)
    centroids = _initialize_centroids(X, k, seed)

    for i in tqdm(range(max_iter)):
        _, labels = _assign_clusters(X, centroids, batch_size)
        new_centroids = _update_centroids(X, labels, centroids)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if shift < eps:  # if shift is small enough
            break

    _, labels = _assign_clusters(X, centroids)
    return (centroids, labels, i + 1)
