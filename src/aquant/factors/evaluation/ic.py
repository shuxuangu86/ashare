from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aquant.factors.operators.cross_sectional import cs_rank


@dataclass(frozen=True, slots=True)
class ICStatistics:
    by_date: tuple[float, ...]
    mean: float
    std: float
    icir: float
    positive_ratio: float
    t_stat: float


def _rank(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(cs_rank(values), dtype=np.float64)


def information_coefficient(
    factor: npt.ArrayLike,
    forward_returns: npt.ArrayLike,
    *,
    rank: bool = False,
) -> ICStatistics:
    values = np.asarray(factor, dtype=np.float64)
    returns = np.asarray(forward_returns, dtype=np.float64)
    if values.ndim != 2 or values.shape != returns.shape:
        raise ValueError("IC inputs must be equal time-by-security matrices")
    by_date: list[float] = []
    for factor_row, return_row in zip(values, returns, strict=True):
        valid = np.isfinite(factor_row) & np.isfinite(return_row)
        if np.count_nonzero(valid) < 2:
            by_date.append(float("nan"))
            continue
        left, right = factor_row[valid], return_row[valid]
        if rank:
            left, right = _rank(left), _rank(right)
        by_date.append(
            float(np.corrcoef(left, right)[0, 1])
            if np.std(left) > 0 and np.std(right) > 0
            else float("nan")
        )
    finite = np.asarray(by_date)[np.isfinite(by_date)]
    mean = float(np.mean(finite)) if len(finite) else float("nan")
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
    return ICStatistics(
        tuple(by_date),
        mean,
        std,
        mean / std if std > 0 else float("nan"),
        float(np.mean(finite > 0)) if len(finite) else float("nan"),
        mean / (std / np.sqrt(len(finite))) if std > 0 else float("nan"),
    )


def forward_return_labels(close: npt.ArrayLike, horizon: int) -> npt.NDArray[np.float64]:
    if horizon <= 0:
        raise ValueError("label horizon must be positive")
    prices = np.asarray(close, dtype=np.float64)
    if prices.ndim != 2:
        raise ValueError("close matrix must be time-by-security")
    result = np.full(prices.shape, np.nan)
    if horizon < len(prices):
        with np.errstate(all="ignore"):
            result[:-horizon] = prices[horizon:] / prices[:-horizon] - 1
    result[~np.isfinite(result)] = np.nan
    return result
