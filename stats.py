import math

import numpy as np


def count_unique_colors(pixels: np.ndarray) -> int:
    """Count the number of distinct RGB colors in a pixel array, shape (N, 3)"""
    return len(np.unique(pixels, axis=0))


def theoretical_size_bytes(n_pixels: int, n_colors: int) -> int:
    """Storage size for the compressed image: palette + per pixel index"""
    palette_bytes = 3 * n_colors
    bits_per_pixel = math.ceil(math.log2(n_colors))
    index_bytes = math.ceil(n_pixels * bits_per_pixel / 8)
    return palette_bytes + index_bytes


def compression_report(
    pixels: np.ndarray, centroids: np.ndarray, labels: np.ndarray
) -> dict:
    """Summarize the compression achieved: colors, sizes, and savings

    Formula: S_compressed = 3K (palette) + (N * ceil(log_2(K)))/ 8 (per pixel index)
    """
    n_pixels = len(pixels)
    original_colors = count_unique_colors(pixels)
    compressed_colors = len(np.unique(labels))  # clusters used

    original_bytes = n_pixels * 3
    compressed_bytes = theoretical_size_bytes(n_pixels, compressed_colors)
    savings = 1 - compressed_bytes / original_bytes

    return {
        "n_pixels": n_pixels,
        "original_unique_colors": original_colors,
        "compressed_unique_colors": compressed_colors,
        "original_size_bytes": original_bytes,
        "compressed_size_bytes": compressed_bytes,
        "savings_percent": savings * 100,
    }
