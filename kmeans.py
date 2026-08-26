from __future__ import annotations

import numpy as np

from image_io import load_pixels  # for testing


def initialize_centroids(X: np.ndarray, k: int, seed: int | None) -> np.ndarray:
    """Initialize an array with k-random starting centroids"""
    rng = np.random.default_rng(seed)
    centroids = rng.choice(X, k, replace=False)
    return centroids


def assign_clusters(X: np.ndarray, centroids: np.ndarray):
    """Assign labels to the nearest centroids creating clusters, returns (squared distance, id of nearest centroid)"""

    n = len(X)
    distances = np.full(n, np.inf)
    labels = np.zeros(n, dtype=int)
    for id_pixel, pixel in enumerate(X):
        for id_c, c in enumerate(centroids):
            val = _squared_distance(c, pixel)
            if val < distances[id_pixel]:
                distances[id_pixel] = val
                labels[id_pixel] = id_c
    return distances, labels


def update_centroids(
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


def _squared_distance(
    centroid: tuple[int, int, int], pixel: tuple[int, int, int]
) -> float:
    x_c, y_c, z_c = centroid
    x_p, y_p, z_p = pixel

    def _distance(a: int, b: int):
        return (a - b) ** 2

    return _distance(x_c, x_p) + _distance(y_c, y_p) + _distance(z_c, z_p)


def fit(
    X: np.array, k: int, seed: int | None = None, max_iter: int = 100, eps: float = 1e-4
):
    """Run k-means clustering algorithm. Returns (centroids, labels, n_iter)"""
    X = X.astype(np.float64)
    centroids = initialize_centroids(X, k, seed)

    for i in range(max_iter):
        _, labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, centroids)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if shift < eps:  # if shift is small enough
            break

    _, labels = assign_clusters(X, centroids)
    return (centroids, labels, i + 1)


if __name__ == "__main__":
    pixels, (height, width) = load_pixels("data/results/test-small.npz")
    print(f"loaded {len(pixels):,} pixels ({height}x{width})")

    k = 8
    centroids, labels, n_iter = fit(pixels, k=k, seed=0)

    print(f"converged in {n_iter} iterations")
    print("centroids (RGB):")
    print(np.round(centroids).astype(int))
    print("cluster sizes:", np.bincount(labels, minlength=k))
