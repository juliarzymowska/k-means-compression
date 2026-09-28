from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import kmeans
import kmeans_sklearn
from image_io import image_load
from kmeans import KMEANS_DEFAULTS
from stats import compression_report


def indexed_image(
    centroids: np.ndarray, labels: np.ndarray, height: int, width: int
) -> Image.Image:
    """Build a palette-mode image: real per-pixel palette indices + a real palette.

    Use PNG encoding to make k-means clustering savings on file size.
    """
    palette_image = Image.new("P", (width, height))
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[: len(centroids)] = np.round(centroids).astype(np.uint8)
    palette_image.putpalette(palette.flatten().tolist())
    palette_image.putdata(labels.astype(np.uint8).tolist())
    return palette_image


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
    if save_path.suffix.lower() != ".png":
        # JPEG can't store palette indices (it's a DCT codec)
        # PNG stores them losslessly, so it's the only format that shows the compression on disk
        save_path = save_path.with_suffix(".png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    im = indexed_image(centroids, labels, height, width)
    im.save(save_path, optimize=True)

    report = compression_report(X, centroids, labels)
    report["save_path"] = str(save_path)
    report["actual_size_bytes"] = save_path.stat().st_size
    return report
