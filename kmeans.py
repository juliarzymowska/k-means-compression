from __future__ import annotations

import numpy as np

from image_io import image_load  # for testing


def _initialize_centroids(X: np.ndarray, k: int, seed: int | None = None) -> np.ndarray:
    """Initialize an array with k-random starting centroids"""
    rng = np.random.default_rng(seed)
    centroids = rng.choice(X, k, replace=False)
    return centroids


def _assign_clusters(X: np.ndarray, centroids: np.ndarray):
    """Assign labels to the nearest centroids creating clusters, returns (squared distance, id of nearest centroid)"""

    distances = (
        (X**2).sum(axis=1).reshape(-1, 1)
        + (centroids**2).sum(axis=1).reshape(1, -1)
        - 2 * np.dot(X, centroids.T)
    )  # sum each row together (a-b)^2 = ||a||^2 + ||b||^2 - 2|a||b|, returns for each pixel k distances to centroids

    labels = np.argmin(
        distances,
        axis=1,
    )
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


def fit(
    X: np.array, k: int, seed: int | None = None, max_iter: int = 100, eps: float = 1e-4
):
    """Run k-means clustering algorithm. Returns (centroids, labels, n_iter)"""
    X = X.astype(np.float64)
    centroids = _initialize_centroids(X, k, seed)

    for i in range(max_iter):
        _, labels = _assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, centroids)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if shift < eps:  # if shift is small enough
            break

    _, labels = _assign_clusters(X, centroids)
    return (centroids, labels, i + 1)


if __name__ == "__main__":
    # pixels, (height, width) = load_pixels("data/results/test-small.npz")
    pixels, (height, width) = image_load("data/test-small.png")
    pixels = pixels.reshape(-1, 3)
    print(f"loaded {len(pixels):,} pixels ({height}x{width})")

    k = 4
    centroids, labels, n_iter = fit(pixels, k=k, seed=0)

    print(f"converged in {n_iter} iterations")
    print("centroids (RGB):")
    print(np.round(centroids).astype(int))
    print("cluster sizes:", np.bincount(labels, minlength=k))
