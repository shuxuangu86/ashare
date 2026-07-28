import numpy as np
import numpy.typing as npt

from aquant.factors.spec import FactorSpec


def exact_deduplicate(specs: tuple[FactorSpec, ...]) -> tuple[FactorSpec, ...]:
    unique: dict[str, FactorSpec] = {}
    for spec in specs:
        unique.setdefault(spec.expression_hash, spec)
    return tuple(unique.values())


def numerical_duplicates(
    values: npt.ArrayLike,
    *,
    correlation_threshold: float = 0.995,
) -> tuple[tuple[int, int], ...]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not 0 < correlation_threshold <= 1:
        raise ValueError("deduplication requires observations-by-factors and valid threshold")
    duplicates: list[tuple[int, int]] = []
    for left in range(matrix.shape[1]):
        for right in range(left + 1, matrix.shape[1]):
            valid = np.isfinite(matrix[:, left]) & np.isfinite(matrix[:, right])
            if np.count_nonzero(valid) < 2:
                continue
            x, y = matrix[valid, left], matrix[valid, right]
            identical = np.array_equal(x, y)
            correlated = (
                np.std(x) > 0
                and np.std(y) > 0
                and abs(np.corrcoef(x, y)[0, 1]) >= correlation_threshold
            )
            if identical or correlated:
                duplicates.append((left, right))
    return tuple(duplicates)
