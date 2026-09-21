import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from error import elbow_max_k_int, non_negative_int
from image_io import image_load
from kmeans import KMEANS_DEFAULTS, fit


def _within_clusters_sum_of_squares(
    X: np.ndarray, centroids: np.ndarray, labels: np.ndarray
) -> float:
    diffs = X - centroids[labels]
    return float(np.sum(diffs**2))


def compute_elbow_curve(
    X: np.ndarray, k_values: list[int], seed: int | None = KMEANS_DEFAULTS["seed"]
) -> np.ndarray:

    wcss = np.zeros_like(k_values, dtype=float)
    for i, k in enumerate(k_values):
        centroids, labels, _ = fit(X, k, seed)
        wcss[i] = _within_clusters_sum_of_squares(X, centroids, labels)

    return wcss


def find_elbow_k(k_values: np.ndarray, wcss: np.ndarray):
    k_values = np.array(k_values, dtype=float)
    wcss = np.array(wcss, dtype=float)

    k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    wcss_norm = (wcss - wcss.min()) / (wcss.max() - wcss.min())

    x_1, y_1 = k_norm[0], wcss_norm[0]
    x_2, y_2 = k_norm[-1], wcss_norm[-1]

    def _herons_formula(a: float, b: float, c: float) -> float:
        s = 0.5 * (a + b + c)
        return np.sqrt(s * (s - a) * (s - b) * (s - c))

    def _distance(x_1: float, y_1: float, x_2: float, y_2: float) -> float:
        return np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

    def _find_height(area: float, base: float) -> float:
        return 2 * area / base

    def _max_height(height: np.ndarray) -> int:
        return np.argmax(height)

    base = _distance(x_1, y_1, x_2, y_2)
    height = np.zeros_like(k_values, dtype=float)

    for i in range(1, len(k_values) - 1):
        x_i, y_i = k_norm[i], wcss_norm[i]
        line_length_1 = _distance(x_1, y_1, x_i, y_i)
        line_length_2 = _distance(x_i, y_i, x_2, y_2)

        area = _herons_formula(line_length_1, line_length_2, base)

        height[i] = _find_height(area, base)

    return k_values[_max_height(height)]


def plot_optimal_k(optimal_k: float, wcss: np.ndarray, k_values: np.ndarray) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss, marker="o")
    plt.plot(
        k_values[optimal_k - 1],  # -1, because we are indexing from 0
        wcss[optimal_k - 1],  # same as above
        marker="D",
        color="green",
        markersize=14,
    )
    plt.xlabel("K")
    plt.ylabel("Inertia (WCSS)")
    plt.title("Elbow method")
    plt.xticks(k_values)
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finds optimal amount of colors (k) for particular image compression."
    )

    parser.add_argument(
        "--load",
        help="path to image that should be compressed",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max_k",
        help="maximum value of k to check, else use 32",
        type=elbow_max_k_int,
        default=17,
        required=False,
    )

    parser.add_argument(
        "--seed", type=non_negative_int, default=KMEANS_DEFAULTS["seed"]
    )
    args = parser.parse_args()

    try:
        K_RANGE = list(range(2, args.max_k))
        pixels, _ = image_load(args.load)
        X = pixels.reshape(-1, 3)
        wcss = compute_elbow_curve(X, K_RANGE, args.seed)
        optimal_k = find_elbow_k(K_RANGE, wcss).astype(int)
        print(f"Optimal k: {optimal_k}\n")
        plot_optimal_k(optimal_k, wcss, K_RANGE)

    except FileNotFoundError:
        print(f"Error: could not find image file: {args.load}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
