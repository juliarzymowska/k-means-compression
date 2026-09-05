import numpy as np

from kmeans import KMEANS_DEFAULTS, fit


def within_clusters_sum_of_squares(
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
        wcss[i] = within_clusters_sum_of_squares(X, centroids, labels)

    return wcss


def find_elbow_k(k_values: np.ndarray, wcss: np.ndarray):
    k_values = np.array(k_values, dtype=float)
    wcss = np.array(wcss, dtype=float)

    k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    wcss_norm = (wcss - wcss.min()) / (wcss.max() - wcss.min())

    x_1, y_1 = k_norm[0], wcss_norm[0]
    x_2, y_2 = k_norm[-1], wcss_norm[-1]

    def herons_formula(a: float, b: float, c: float) -> float:
        s = 0.5 * (a + b + c)
        return np.sqrt(s * (s - a) * (s - b) * (s - c))

    def distance(x_1: float, y_1: float, x_2: float, y_2: float) -> float:
        return np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

    def find_height(area: float, base: float) -> float:
        return 2 * area / base

    def max_height(height: np.ndarray) -> int:
        return np.argmax(height)

    base = distance(x_1, y_1, x_2, y_2)
    height = np.zeros_like(k_values, dtype=float)

    for i in range(1, len(k_values) - 1):
        x_i, y_i = k_norm[i], wcss_norm[i]
        line_length_1 = distance(x_1, y_1, x_i, y_i)
        line_length_2 = distance(x_i, y_i, x_2, y_2)

        area = herons_formula(line_length_1, line_length_2, base)

        height[i] = find_height(area, base)

    return k_values[max_height(height)]
