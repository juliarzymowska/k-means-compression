from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import kmeans
from image_io import image_load


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
    seed: int | None,
    max_iter: int = 100,
    eps: float = 1e-4,
    batch_size: int = 100_000,
):
    image, (height, width) = image_load(load_path)
    X = image.reshape(-1, 3)

    centroids, labels, _ = kmeans.fit(X, k, seed, max_iter, eps, batch_size)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    compressed_image = image_reconstruction(centroids, labels, height, width)
    im = Image.fromarray(compressed_image)
    im.save(save_path)


if __name__ == "__main__":
    pipeline("data/test-small.png", "data/results/test-small.png", k=4, seed=1)
