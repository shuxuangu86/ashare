from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from math import floor

import numpy as np
import numpy.typing as npt

from aquant.factors.evaluation.analyzer import FactorAnalyzer
from aquant.factors.evaluation.ic import information_coefficient
from aquant.factors.operators.time_series import rolling_cov, rolling_std, rolling_var

Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class InstitutionalEvaluation:
    horizon: int
    pearson_ic_mean: float
    rank_ic_mean: float
    icir: float
    rank_icir: float
    newey_west_t: float
    ic_positive_ratio: float
    annual_rank_ic: tuple[tuple[int, float], ...]
    quarterly_rank_ic: tuple[tuple[str, float], ...]
    regime_rank_ic: tuple[tuple[str, float], ...]
    quantile_returns: tuple[float, ...]
    monotonicity: float
    long_short_return: float
    turnover: float
    net_long_short_return: float
    max_drawdown: float
    style_exposures: tuple[tuple[str, float], ...]
    observations: int

    def payload(self) -> dict[str, object]:
        return asdict(self)


def evaluate_institutional(
    factor_values: npt.ArrayLike,
    forward_returns: npt.ArrayLike,
    trade_dates: tuple[date, ...],
    *,
    horizon: int,
    quantiles: int = 5,
    cost_bps: float = 10,
    regimes: tuple[str, ...] | None = None,
    exposures: Mapping[str, npt.ArrayLike] | None = None,
) -> InstitutionalEvaluation:
    factor = np.asarray(factor_values, dtype=np.float64)
    returns = np.asarray(forward_returns, dtype=np.float64)
    if factor.shape != returns.shape or factor.ndim != 2 or len(trade_dates) != len(factor):
        raise ValueError("institutional evaluation inputs must align")
    if horizon <= 0 or quantiles < 2 or cost_bps < 0:
        raise ValueError("institutional evaluation parameters are invalid")
    if regimes is not None and len(regimes) != len(trade_dates):
        raise ValueError("market regimes must align with dates")
    base = FactorAnalyzer().evaluate(factor, returns, quantiles=quantiles)
    pearson = information_coefficient(factor, returns)
    ranked = information_coefficient(factor, returns, rank=True)
    rank_series = np.asarray(ranked.by_date)
    long_short = long_short_return_series(factor, returns, quantiles)
    net = long_short - base.turnover * cost_bps / 10_000
    finite_net = net[np.isfinite(net)]
    style = tuple(
        sorted(
            (
                name,
                _mean_cross_sectional_correlation(factor, np.asarray(values, dtype=float)),
            )
            for name, values in (exposures or {}).items()
        )
    )
    return InstitutionalEvaluation(
        horizon=horizon,
        pearson_ic_mean=pearson.mean,
        rank_ic_mean=ranked.mean,
        icir=pearson.icir,
        rank_icir=ranked.icir,
        newey_west_t=newey_west_mean_t(rank_series),
        ic_positive_ratio=ranked.positive_ratio,
        annual_rank_ic=_annual_means(trade_dates, rank_series),
        quarterly_rank_ic=_quarterly_means(trade_dates, rank_series),
        regime_rank_ic=_regime_means(regimes, rank_series),
        quantile_returns=base.quantile_returns,
        monotonicity=base.monotonicity,
        long_short_return=base.long_short_return,
        turnover=base.turnover,
        net_long_short_return=(float(np.mean(finite_net)) if len(finite_net) else float("nan")),
        max_drawdown=_max_drawdown(net),
        style_exposures=style,
        observations=int(np.count_nonzero(np.isfinite(factor) & np.isfinite(returns))),
    )


def newey_west_mean_t(values: npt.ArrayLike, max_lag: int | None = None) -> float:
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    count = len(sample)
    if count < 2:
        return float("nan")
    centered = sample - np.mean(sample)
    lag_limit = (
        min(count - 1, floor(4 * (count / 100) ** (2 / 9)))
        if max_lag is None
        else min(count - 1, max_lag)
    )
    if lag_limit < 0:
        raise ValueError("Newey-West lag cannot be negative")
    long_run_variance = float(centered @ centered / count)
    for lag in range(1, lag_limit + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / count)
        long_run_variance += 2 * (1 - lag / (lag_limit + 1)) * covariance
    standard_error = np.sqrt(max(long_run_variance, 0) / count)
    return float(np.mean(sample) / standard_error) if standard_error > 0 else float("nan")


def basic_style_exposures(
    close: npt.ArrayLike,
    float_market_cap: npt.ArrayLike,
    turnover_rate: npt.ArrayLike,
    *,
    window: int = 60,
) -> dict[str, Array]:
    prices = np.asarray(close, dtype=float)
    market_cap = np.asarray(float_market_cap, dtype=float)
    turnover = np.asarray(turnover_rate, dtype=float)
    if prices.shape != market_cap.shape or prices.shape != turnover.shape or prices.ndim != 2:
        raise ValueError("style inputs must be aligned time-by-security matrices")
    if window < 2:
        raise ValueError("style exposure window must be at least two")
    with np.errstate(all="ignore"):
        returns = prices / np.roll(prices, 1, axis=0) - 1
        size = np.log(market_cap)
    returns[0] = np.nan
    market_returns = np.full(len(returns), np.nan)
    if len(returns) > 1:
        market_returns[1:] = np.nanmean(returns[1:], axis=1)
    volatility = rolling_std(returns, window, ddof=1)
    market_matrix = np.broadcast_to(market_returns[:, None], returns.shape)
    market_variance = rolling_var(market_returns, window, ddof=1)
    with np.errstate(all="ignore"):
        beta = rolling_cov(returns, market_matrix, window, ddof=1) / market_variance[:, None]
    beta[~np.isfinite(beta)] = np.nan
    return {
        "beta": beta,
        "liquidity": turnover,
        "price": prices,
        "size": size,
        "volatility": volatility,
    }


def historical_market_regimes(close: npt.ArrayLike, window: int = 20) -> tuple[str, ...]:
    prices = np.asarray(close, dtype=float)
    if prices.ndim != 2 or window <= 0:
        raise ValueError("market regime input is invalid")
    market = np.nanmean(prices, axis=1)
    regimes: list[str] = []
    for index in range(len(market)):
        if index < window or not np.isfinite(market[index - window : index + 1]).all():
            regimes.append("UNCLASSIFIED")
            continue
        change = market[index] / market[index - window] - 1
        regimes.append("BULL" if change > 0.05 else "BEAR" if change < -0.05 else "SIDEWAYS")
    return tuple(regimes)


def _annual_means(
    dates: tuple[date, ...],
    values: Array,
) -> tuple[tuple[int, float], ...]:
    grouped: dict[int, list[float]] = {}
    for value_date, value in zip(dates, values, strict=True):
        if np.isfinite(value):
            grouped.setdefault(value_date.year, []).append(float(value))
    return tuple((key, float(np.mean(grouped[key]))) for key in sorted(grouped))


def _quarterly_means(
    dates: tuple[date, ...],
    values: Array,
) -> tuple[tuple[str, float], ...]:
    grouped: dict[str, list[float]] = {}
    for value_date, value in zip(dates, values, strict=True):
        if np.isfinite(value):
            key = f"{value_date.year}Q{(value_date.month - 1) // 3 + 1}"
            grouped.setdefault(key, []).append(float(value))
    return tuple((key, float(np.mean(grouped[key]))) for key in sorted(grouped))


def _regime_means(
    regimes: tuple[str, ...] | None,
    values: Array,
) -> tuple[tuple[str, float], ...]:
    if regimes is None:
        return ()
    grouped: dict[str, list[float]] = {}
    for regime, value in zip(regimes, values, strict=True):
        if np.isfinite(value):
            grouped.setdefault(regime, []).append(float(value))
    return tuple((key, float(np.mean(grouped[key]))) for key in sorted(grouped))


def long_short_return_series(
    factor_values: npt.ArrayLike,
    forward_returns: npt.ArrayLike,
    quantiles: int = 5,
) -> Array:
    factor = np.asarray(factor_values, dtype=float)
    returns = np.asarray(forward_returns, dtype=float)
    if factor.shape != returns.shape or factor.ndim != 2 or quantiles < 2:
        raise ValueError("long-short inputs must be aligned matrices and valid quantiles")
    result = np.full(len(factor), np.nan)
    for index, (factor_row, return_row) in enumerate(zip(factor, returns, strict=True)):
        valid = np.isfinite(factor_row) & np.isfinite(return_row)
        if np.count_nonzero(valid) < quantiles:
            continue
        order = np.argsort(factor_row[valid], kind="stable")
        ordered_returns = return_row[valid][order]
        group_size = len(ordered_returns) // quantiles
        result[index] = np.mean(ordered_returns[-group_size:]) - np.mean(
            ordered_returns[:group_size]
        )
    return result


def _max_drawdown(returns: Array) -> float:
    finite = returns[np.isfinite(returns)]
    if not len(finite):
        return float("nan")
    wealth = np.cumprod(1 + finite)
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1))


def _mean_cross_sectional_correlation(factor: Array, exposure: Array) -> float:
    if factor.shape != exposure.shape:
        raise ValueError("style exposure matrix must align with factor")
    correlations: list[float] = []
    for left, right in zip(factor, exposure, strict=True):
        valid = np.isfinite(left) & np.isfinite(right)
        if np.count_nonzero(valid) > 1 and np.std(left[valid]) and np.std(right[valid]):
            correlations.append(float(np.corrcoef(left[valid], right[valid])[0, 1]))
    return float(np.mean(correlations)) if correlations else float("nan")
