from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aquant.factors.operators.cross_sectional import cs_rank


@dataclass(frozen=True, slots=True)
class StateForwardEvaluation:
    state_id: str
    target_id: str
    horizon: int
    observations: int
    pearson_correlation: float | None
    spearman_correlation: float | None
    bottom_decile_mean: float | None
    lower_middle_mean: float | None
    middle_mean: float | None
    upper_middle_mean: float | None
    top_decile_mean: float | None
    conditional_spread: float | None
    newey_west_t: float | None
    direction_hit_rate: float | None


def forward_returns(level: npt.NDArray[np.float64], horizon: int) -> npt.NDArray[np.float64]:
    if horizon <= 0:
        raise ValueError("forward horizon must be positive")
    output = np.full(level.shape, np.nan)
    valid = np.isfinite(level[:-horizon]) & np.isfinite(level[horizon:]) & (level[:-horizon] > 0)
    output[:-horizon][valid] = level[horizon:][valid] / level[:-horizon][valid] - 1.0
    return output


def evaluate_state(
    state_id: str,
    state: npt.NDArray[np.float64],
    future_return: npt.NDArray[np.float64],
    *,
    horizon: int,
    target_id: str = "ALL_A",
) -> StateForwardEvaluation:
    if state.shape != future_return.shape:
        raise ValueError("state and forward return shape mismatch")
    valid = np.isfinite(state) & np.isfinite(future_return)
    count = int(np.sum(valid))
    if count < 20:
        return StateForwardEvaluation(
            state_id,
            target_id,
            horizon,
            count,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    x = state[valid]
    y = future_return[valid]
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
    x_rank = _rank(x)
    y_rank = _rank(y)
    spearman = (
        float(np.corrcoef(x_rank, y_rank)[0, 1])
        if np.std(x_rank) > 0 and np.std(y_rank) > 0
        else None
    )
    q10, q30, q70, q90 = np.quantile(x, (0.1, 0.3, 0.7, 0.9))
    bottom = _mean_or_none(y[x <= q10])
    top = _mean_or_none(y[x >= q90])
    spread = top - bottom if top is not None and bottom is not None else None
    centered = x - np.median(x)
    return StateForwardEvaluation(
        state_id,
        target_id,
        horizon,
        count,
        pearson,
        spearman,
        bottom,
        _mean_or_none(y[(x > q10) & (x <= q30)]),
        _mean_or_none(y[(x > q30) & (x < q70)]),
        _mean_or_none(y[(x >= q70) & (x < q90)]),
        top,
        spread,
        _newey_west_t(centered * y, max_lag=max(1, horizon - 1)),
        float(np.mean((centered * y) > 0)),
    )


def correlation_matrix(
    states: dict[str, npt.NDArray[np.float64]],
    *,
    minimum_observations: int = 60,
) -> tuple[tuple[str, str, int, float], ...]:
    if minimum_observations < 2:
        raise ValueError("minimum observations must be at least two")
    names = sorted(states)
    output: list[tuple[str, str, int, float]] = []
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index:]:
            left, right = states[left_name], states[right_name]
            if left.shape != right.shape:
                raise ValueError("state correlation inputs must have equal shape")
            valid = np.isfinite(left) & np.isfinite(right)
            count = int(np.sum(valid))
            if count < minimum_observations:
                continue
            correlation = float(np.corrcoef(left[valid], right[valid])[0, 1])
            if np.isfinite(correlation):
                output.append((left_name, right_name, count, correlation))
    return tuple(output)


def _rank(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(cs_rank(values), dtype=np.float64)


def _mean_or_none(values: npt.NDArray[np.float64]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _newey_west_t(values: npt.NDArray[np.float64], *, max_lag: int) -> float | None:
    count = len(values)
    if count < 20:
        return None
    centered = values - np.mean(values)
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, min(max_lag, count - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1)
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0:
        return None
    return float(np.mean(values) / np.sqrt(long_run_variance / count))
