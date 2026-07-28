from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata  # type: ignore[import-untyped]

from aquant.factors.evaluation.ic import information_coefficient
from aquant.factors.selection.clustering import (
    hierarchical_clusters,
    representative_by_score,
    residual_information,
)

Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ConvergedFactor:
    factor_id: str
    cluster_id: int
    representative_factor_id: str
    is_representative: bool
    within_cluster_correlation: float
    marginal_rank_ic: float
    conditional_rank_ic: float
    incremental_predictive_power: float


@dataclass(frozen=True, slots=True)
class ConvergenceResult:
    factor_ids: tuple[str, ...]
    value_spearman: Array
    long_short_correlation: Array
    rank_ic_correlation: Array
    combined_correlation: Array
    factors: tuple[ConvergedFactor, ...]


def converge_factors(
    factor_values: dict[str, npt.ArrayLike],
    forward_returns: npt.ArrayLike,
    long_short_returns: dict[str, npt.ArrayLike],
    rank_ic_series: dict[str, npt.ArrayLike],
    scores: dict[str, float],
    *,
    maximum_distance: float = 0.3,
) -> ConvergenceResult:
    factor_ids = tuple(sorted(factor_values))
    if (
        len(factor_ids) < 2
        or set(long_short_returns) != set(factor_ids)
        or set(rank_ic_series) != set(factor_ids)
        or set(scores) != set(factor_ids)
    ):
        raise ValueError("convergence inputs must contain the same factors")
    arrays = {key: np.asarray(value, dtype=float) for key, value in factor_values.items()}
    shapes = {value.shape for value in arrays.values()}
    returns = np.asarray(forward_returns, dtype=float)
    if len(shapes) != 1 or next(iter(shapes)) != returns.shape or returns.ndim != 2:
        raise ValueError("factor values and forward returns must align")
    value_spearman = cross_sectional_spearman(arrays)
    long_short = _series_correlation(factor_ids, long_short_returns)
    rank_ic = _series_correlation(factor_ids, rank_ic_series)
    combined = np.mean(
        np.stack((np.abs(value_spearman), np.abs(long_short), np.abs(rank_ic))),
        axis=0,
    )
    np.fill_diagonal(combined, 1)
    clusters = hierarchical_clusters(combined, maximum_distance=maximum_distance)
    representatives = representative_by_score(
        clusters,
        np.asarray([scores[factor_id] for factor_id in factor_ids]),
    )
    resolved: list[ConvergedFactor] = []
    for index, factor_id in enumerate(factor_ids):
        cluster = clusters[index]
        representative_index = representatives[cluster]
        representative_id = factor_ids[representative_index]
        representative_values = arrays[representative_id]
        candidate_values = arrays[factor_id]
        conditional_values = _cross_sectional_residuals(
            candidate_values,
            representative_values,
        )
        candidate_ic = information_coefficient(candidate_values, returns, rank=True).mean
        representative_ic = information_coefficient(
            representative_values,
            returns,
            rank=True,
        ).mean
        conditional_ic = (
            candidate_ic
            if index == representative_index
            else information_coefficient(conditional_values, returns, rank=True).mean
        )
        members = [position for position, label in enumerate(clusters) if label == cluster]
        correlations = [abs(value_spearman[index, member]) for member in members if member != index]
        resolved.append(
            ConvergedFactor(
                factor_id=factor_id,
                cluster_id=cluster,
                representative_factor_id=representative_id,
                is_representative=index == representative_index,
                within_cluster_correlation=(float(np.mean(correlations)) if correlations else 0.0),
                marginal_rank_ic=candidate_ic - representative_ic,
                conditional_rank_ic=conditional_ic,
                incremental_predictive_power=abs(conditional_ic),
            )
        )
    return ConvergenceResult(
        factor_ids,
        value_spearman,
        long_short,
        rank_ic,
        combined,
        tuple(resolved),
    )


def cross_sectional_spearman(factor_values: dict[str, Array]) -> Array:
    """Average daily Spearman correlation after ranking each available cross-section."""
    factor_ids = tuple(sorted(factor_values))
    arrays = tuple(np.asarray(factor_values[factor_id], dtype=float) for factor_id in factor_ids)
    if not arrays or len({array.shape for array in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError("factor values must be aligned time-by-security matrices")
    factor_count = len(factor_ids)
    total = np.zeros((factor_count, factor_count), dtype=float)
    observations = np.zeros((factor_count, factor_count), dtype=np.int64)
    for date_index in range(arrays[0].shape[0]):
        values = np.stack([array[date_index] for array in arrays])
        valid = np.isfinite(values)
        centered_ranks = np.zeros(values.shape, dtype=float)
        for factor_index, row in enumerate(values):
            row_valid = valid[factor_index]
            if np.count_nonzero(row_valid) > 1:
                ranks = rankdata(row[row_valid])
                centered_ranks[factor_index, row_valid] = ranks - np.mean(ranks)
        numerator = centered_ranks @ centered_ranks.T
        left_sum_squares = (centered_ranks * centered_ranks) @ valid.T
        denominator = np.sqrt(left_sum_squares * left_sum_squares.T)
        overlap = valid.astype(np.int64) @ valid.T
        usable = (overlap > 1) & (denominator > 0)
        daily = np.zeros_like(total)
        np.divide(numerator, denominator, out=daily, where=usable)
        total += daily
        observations += usable
    result = np.zeros_like(total)
    np.divide(total, observations, out=result, where=observations > 0)
    np.fill_diagonal(result, 1)
    return result


def _series_correlation(
    factor_ids: tuple[str, ...],
    values: dict[str, npt.ArrayLike],
) -> Array:
    series = [np.asarray(values[factor_id], dtype=float) for factor_id in factor_ids]
    if len({item.shape for item in series}) != 1 or series[0].ndim != 1:
        raise ValueError("factor performance series must be aligned one-dimensional arrays")
    result = np.eye(len(factor_ids))
    for left in range(len(factor_ids)):
        for right in range(left + 1, len(factor_ids)):
            valid = np.isfinite(series[left]) & np.isfinite(series[right])
            correlation = (
                float(np.corrcoef(series[left][valid], series[right][valid])[0, 1])
                if np.count_nonzero(valid) > 1
                and np.std(series[left][valid])
                and np.std(series[right][valid])
                else 0.0
            )
            result[left, right] = result[right, left] = correlation
    return result


def _cross_sectional_residuals(candidate: Array, representative: Array) -> Array:
    result = np.full(candidate.shape, np.nan)
    for index, (left, right) in enumerate(zip(candidate, representative, strict=True)):
        result[index] = residual_information(left, right)
    return result
