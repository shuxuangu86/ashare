import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def _rows(values: npt.ArrayLike) -> tuple[Array, bool]:
    resolved = np.asarray(values, dtype=np.float64)
    if resolved.ndim == 1:
        return resolved[None, :], True
    if resolved.ndim != 2:
        raise ValueError("cross-sectional operators require 1D or 2D arrays")
    return resolved, False


def _restore(values: Array, squeezed: bool) -> Array:
    return values[0] if squeezed else values


def _average_ranks(values: Array) -> Array:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def cs_rank(values: npt.ArrayLike) -> Array:
    rows, squeezed = _rows(values)
    result = np.full(rows.shape, np.nan)
    for index, row in enumerate(rows):
        valid = np.isfinite(row)
        if np.any(valid):
            result[index, valid] = _average_ranks(row[valid]) + 1
    return _restore(result, squeezed)


def cs_percentile(values: npt.ArrayLike) -> Array:
    ranks = cs_rank(values)
    rows, squeezed = _rows(ranks)
    result = np.full(rows.shape, np.nan)
    for index, row in enumerate(rows):
        valid = np.isfinite(row)
        count = np.count_nonzero(valid)
        if count:
            result[index, valid] = (row[valid] - 0.5) / count
    return _restore(result, squeezed)


def cs_demean(values: npt.ArrayLike) -> Array:
    rows, squeezed = _rows(values)
    valid = np.isfinite(rows)
    counts = np.count_nonzero(valid, axis=1)
    sums = np.sum(np.where(valid, rows, 0.0), axis=1)
    means = np.full(len(rows), np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    result = rows - means[:, None]
    result[~np.isfinite(rows)] = np.nan
    return _restore(result, squeezed)


def cs_zscore(values: npt.ArrayLike, *, ddof: int = 0) -> Array:
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    rows, squeezed = _rows(values)
    result = np.full(rows.shape, np.nan)
    for index, row in enumerate(rows):
        valid = np.isfinite(row)
        sample = row[valid]
        if len(sample) <= ddof:
            continue
        standard_deviation = np.std(sample, ddof=ddof)
        result[index, valid] = (
            0.0
            if np.isclose(standard_deviation, 0)
            else (sample - np.mean(sample)) / standard_deviation
        )
    return _restore(result, squeezed)


def winsorize_quantile(
    values: npt.ArrayLike,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> Array:
    if not 0 <= lower < upper <= 1:
        raise ValueError("winsorization quantiles must be ordered in [0, 1]")
    rows, squeezed = _rows(values)
    result = rows.copy()
    for index, row in enumerate(rows):
        valid = np.isfinite(row)
        if np.any(valid):
            limits = np.quantile(row[valid], (lower, upper))
            result[index, valid] = np.clip(row[valid], limits[0], limits[1])
    result[~np.isfinite(rows)] = np.nan
    return _restore(result, squeezed)


def winsorize_mad(values: npt.ArrayLike, *, scale: float = 3.0) -> Array:
    if scale <= 0:
        raise ValueError("winsorization scale must be positive")
    rows, squeezed = _rows(values)
    result = rows.copy()
    for index, row in enumerate(rows):
        valid = np.isfinite(row)
        sample = row[valid]
        if not len(sample):
            continue
        median = np.median(sample)
        mad = np.median(np.abs(sample - median))
        if mad > 0:
            sigma = 1.4826 * mad
            result[index, valid] = np.clip(sample, median - scale * sigma, median + scale * sigma)
    result[~np.isfinite(rows)] = np.nan
    return _restore(result, squeezed)


def _group_apply(values: npt.ArrayLike, groups: npt.ArrayLike, *, zscore: bool) -> Array:
    rows, squeezed = _rows(values)
    resolved_groups = np.asarray(groups, dtype=object)
    if resolved_groups.ndim == 1:
        resolved_groups = np.broadcast_to(resolved_groups, rows.shape)
    if resolved_groups.shape != rows.shape:
        raise ValueError("group labels must match each cross-section")
    result = np.full(rows.shape, np.nan)
    for index, row in enumerate(rows):
        labels = resolved_groups[index]
        for group in {item for item in labels if item is not None}:
            members = (labels == group) & np.isfinite(row)
            sample = row[members]
            if not len(sample):
                continue
            if zscore:
                standard_deviation = np.std(sample)
                result[index, members] = (
                    0.0
                    if np.isclose(standard_deviation, 0)
                    else (sample - np.mean(sample)) / standard_deviation
                )
            else:
                result[index, members] = (_average_ranks(sample) + 0.5) / len(sample)
    return _restore(result, squeezed)


def group_rank(values: npt.ArrayLike, groups: npt.ArrayLike) -> Array:
    return _group_apply(values, groups, zscore=False)


def group_zscore(values: npt.ArrayLike, groups: npt.ArrayLike) -> Array:
    return _group_apply(values, groups, zscore=True)


from aquant.factors.operators.neutralization import (  # noqa: E402
    industry_neutralize,
    multi_exposure_neutralize,
    size_neutralize,
)

__all__ = [
    "cs_demean",
    "cs_percentile",
    "cs_rank",
    "cs_zscore",
    "group_rank",
    "group_zscore",
    "industry_neutralize",
    "multi_exposure_neutralize",
    "size_neutralize",
    "winsorize_mad",
    "winsorize_quantile",
]
