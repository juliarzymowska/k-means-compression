from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import kmeans
import kmeans_sklearn
from image_io import image_load
from kmeans import KMEANS_DEFAULTS
from stats import compression_report


def image_reconstruction(
    centroids: np.ndarray, labels: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Rebuild the compressed image: every pixel becomes its cluster's centroid color."""
    compressed_pixels = centroids[labels]
    compressed_image = compressed_pixels.reshape(
        height, width, 3
    )  # back to a real image size
    return np.round(compressed_image).astype(
        np.uint8
    )  # uint8 is a type for images (0, 255) :)


def pipeline(
    load_path: Path,
    save_path: Path,
    k: int,
    backend: str = "scratch",
    seed: int | None = KMEANS_DEFAULTS["seed"],
    max_iter: int = KMEANS_DEFAULTS["max_iter"],
    eps: float = KMEANS_DEFAULTS["eps"],
    batch_size: int = KMEANS_DEFAULTS["batch_size"],
) -> dict:
    image, (height, width) = image_load(load_path)
    X = image.reshape(-1, 3)

    if k > len(X):
        raise ValueError(
            f"k={k} exceeds the number of pixels in the image ({len(X):,})"
        )

    if backend == "scratch":
        centroids, labels, n_iter = kmeans.fit(X, k, seed, max_iter, eps)
    elif backend == "sklearn":
        centroids, labels, n_iter = kmeans_sklearn.fit(X, k, seed, max_iter, eps)  # noqa: RUF059
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    compressed_image = image_reconstruction(centroids, labels, height, width)
    im = Image.fromarray(compressed_image)
    im.save(save_path)
    return compression_report(X, centroids, labels)
