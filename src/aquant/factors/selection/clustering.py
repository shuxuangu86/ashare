import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore[import-untyped]
from scipy.spatial.distance import squareform  # type: ignore[import-untyped]


def correlation_matrix(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("clustering requires observations-by-factors")
    result = np.corrcoef(matrix, rowvar=False)
    return np.nan_to_num(result, nan=0.0)


def hierarchical_clusters(
    correlations: npt.ArrayLike,
    *,
    maximum_distance: float = 0.3,
) -> tuple[int, ...]:
    matrix = np.asarray(correlations, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("correlation matrix must be square")
    distance = 1 - np.abs(matrix)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=True), method="average")
    return tuple(int(item) for item in fcluster(tree, maximum_distance, criterion="distance"))


def representative_by_score(
    clusters: tuple[int, ...],
    scores: npt.ArrayLike,
) -> dict[int, int]:
    resolved = np.asarray(scores, dtype=float)
    if len(clusters) != len(resolved):
        raise ValueError("cluster labels and scores must align")
    return {
        cluster: max(
            (index for index, label in enumerate(clusters) if label == cluster),
            key=lambda index: resolved[index],
        )
        for cluster in sorted(set(clusters))
    }


def residual_information(
    candidate: npt.ArrayLike, representative: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    y, x = np.asarray(candidate, dtype=float), np.asarray(representative, dtype=float)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("residual information inputs must be aligned vectors")
    valid = np.isfinite(y) & np.isfinite(x)
    result = np.full(y.shape, np.nan)
    design = np.column_stack((np.ones(np.count_nonzero(valid)), x[valid]))
    coefficients = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    result[valid] = y[valid] - design @ coefficients
    return result
