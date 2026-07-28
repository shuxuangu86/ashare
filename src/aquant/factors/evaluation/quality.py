from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class FactorQuality:
    coverage: float
    missing_rate: float
    valid_count: int
    unique_count: int
    zero_rate: float
    mean: float
    std: float
    minimum: float
    maximum: float
    quantiles: tuple[float, ...]
    skew: float
    kurtosis: float
    infinite_count: int
    cross_sectional_dispersion: float
    date_coverage: float
    security_coverage: float
    largest_gap: int
    flags: tuple[str, ...]


def evaluate_quality(values: npt.ArrayLike) -> FactorQuality:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError("factor quality requires a non-empty time-by-security matrix")
    finite = np.isfinite(matrix)
    sample = matrix[finite]
    valid_count = int(sample.size)
    coverage = valid_count / matrix.size
    date_valid = np.any(finite, axis=1)
    security_valid = np.any(finite, axis=0)
    gaps: list[int] = []
    current = 0
    for valid in date_valid:
        current = 0 if valid else current + 1
        gaps.append(current)
    flags: list[str] = []
    if valid_count == 0:
        flags.append("ALL_NULL")
    unique_count = len(np.unique(sample)) if valid_count else 0
    standard_deviation = float(np.std(sample)) if valid_count else float("nan")
    if valid_count and (unique_count <= 1 or np.isclose(standard_deviation, 0)):
        flags.append("NEAR_CONSTANT")
    if coverage < 0.5:
        flags.append("LOW_COVERAGE")
    row_dispersion = [
        float(np.std(row[valid]))
        for row, valid in zip(matrix, finite, strict=True)
        if np.any(valid)
    ]
    dispersion = float(np.median(row_dispersion)) if row_dispersion else float("nan")
    if valid_count:
        mean = float(np.mean(sample))
        centered = sample - mean
        sigma = np.std(sample)
        skew = float(np.mean((centered / sigma) ** 3)) if sigma > 0 else 0.0
        kurtosis = float(np.mean((centered / sigma) ** 4) - 3) if sigma > 0 else 0.0
        quantiles = tuple(
            float(item) for item in np.quantile(sample, (0.01, 0.25, 0.5, 0.75, 0.99))
        )
    else:
        mean = skew = kurtosis = float("nan")
        quantiles = (float("nan"),) * 5
    return FactorQuality(
        coverage=coverage,
        missing_rate=1 - coverage,
        valid_count=valid_count,
        unique_count=unique_count,
        zero_rate=float(np.mean(sample == 0)) if valid_count else float("nan"),
        mean=mean,
        std=standard_deviation,
        minimum=float(np.min(sample)) if valid_count else float("nan"),
        maximum=float(np.max(sample)) if valid_count else float("nan"),
        quantiles=quantiles,
        skew=skew,
        kurtosis=kurtosis,
        infinite_count=int(np.count_nonzero(np.isinf(matrix))),
        cross_sectional_dispersion=float(dispersion),
        date_coverage=float(np.mean(date_valid)),
        security_coverage=float(np.mean(security_valid)),
        largest_gap=max(gaps, default=0),
        flags=tuple(flags),
    )
