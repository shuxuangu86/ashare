from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import numpy.typing as npt

from aquant.strategies.microcap.experiments import (
    RebalanceFrequency,
    generate_rebalance_dates,
)

FloatArray = npt.NDArray[np.float64]


def select_l4_configuration(
    *,
    scores: npt.ArrayLike,
    close: npt.ArrayLike,
    trade_dates: tuple[date, ...],
    validation_positions: npt.NDArray[np.int64],
    target_counts: tuple[int, ...] = (30, 50, 80, 100),
    frequencies: tuple[RebalanceFrequency, ...] = (
        RebalanceFrequency.WEEKLY,
        RebalanceFrequency.MONTHLY,
    ),
    cost_bps: float = 10.0,
) -> dict[str, Any]:
    """Choose an L4 configuration on one inner validation window only."""
    values = np.asarray(scores, dtype=np.float64)
    prices = np.asarray(close, dtype=np.float64)
    if values.shape != prices.shape or values.ndim != 2:
        raise ValueError("L4 scores and close must be aligned date-by-security matrices")
    if len(trade_dates) != len(prices) or len(validation_positions) < 20:
        raise ValueError("L4 validation dates are missing or insufficient")
    if np.any(np.diff(validation_positions) != 1):
        raise ValueError("L4 validation positions must be contiguous and ordered")
    if not target_counts or any(count <= 0 for count in target_counts):
        raise ValueError("L4 target counts must be positive")
    if not frequencies or cost_bps < 0:
        raise ValueError("L4 frequencies and costs are invalid")
    candidates = [
        _simulate(
            scores=values,
            close=prices,
            trade_dates=trade_dates,
            positions=validation_positions,
            target_count=target_count,
            frequency=frequency,
            cost_bps=cost_bps,
        )
        for target_count in target_counts
        for frequency in frequencies
    ]
    selected = max(
        candidates,
        key=lambda item: (
            float(item["selection_score"]),
            -float(item["annual_turnover"]),
            -int(item["target_count"]),
        ),
    )
    return {
        "selection_window_start": trade_dates[int(validation_positions[0])].isoformat(),
        "selection_window_end": trade_dates[int(validation_positions[-1])].isoformat(),
        "cost_bps": cost_bps,
        "outer_test_used_for_selection": False,
        "objective": "annual_excess_return_plus_0.05_times_excess_sharpe",
        "selected": selected,
        "candidates": candidates,
    }


def _simulate(
    *,
    scores: FloatArray,
    close: FloatArray,
    trade_dates: tuple[date, ...],
    positions: npt.NDArray[np.int64],
    target_count: int,
    frequency: RebalanceFrequency,
    cost_bps: float,
) -> dict[str, Any]:
    dates = tuple(trade_dates[int(position)] for position in positions)
    rebalance_dates = generate_rebalance_dates(dates, frequency)
    active = np.zeros(close.shape[1], dtype=np.float64)
    pending: FloatArray | None = None
    strategy_returns: list[float] = []
    benchmark_returns: list[float] = []
    total_turnover = 0.0
    for position in positions:
        index = int(position)
        asset_returns = (
            np.divide(
                close[index],
                close[index - 1],
                out=np.full(close.shape[1], np.nan),
                where=np.isfinite(close[index])
                & np.isfinite(close[index - 1])
                & (close[index - 1] > 0),
            )
            - 1
        )
        valid_returns = np.isfinite(asset_returns)
        portfolio_return = float(np.sum(active[valid_returns] * asset_returns[valid_returns]))
        benchmark_return = (
            float(np.mean(asset_returns[valid_returns])) if np.any(valid_returns) else 0.0
        )
        if pending is not None:
            turnover = float(np.sum(np.abs(pending - active)))
            portfolio_return -= turnover * cost_bps / 10_000
            total_turnover += turnover
            active[:] = pending
            pending = None
        strategy_returns.append(portfolio_return)
        benchmark_returns.append(benchmark_return)
        if trade_dates[index] in rebalance_dates:
            row = scores[index]
            eligible = np.flatnonzero(np.isfinite(row) & np.isfinite(close[index]))
            selected = eligible[np.argsort(row[eligible], kind="stable")[-target_count:]]
            pending = np.zeros(close.shape[1], dtype=np.float64)
            if len(selected):
                pending[selected] = 1 / len(selected)
    strategy = np.asarray(strategy_returns)
    benchmark = np.asarray(benchmark_returns)
    excess = strategy - benchmark
    observations = len(strategy)
    annual_strategy = _annualized(strategy)
    annual_benchmark = _annualized(benchmark)
    excess_standard_deviation = float(np.std(excess, ddof=1))
    excess_sharpe = (
        float(np.mean(excess) / excess_standard_deviation * math.sqrt(252))
        if excess_standard_deviation > 0
        else 0.0
    )
    annual_excess = annual_strategy - annual_benchmark
    return {
        "target_count": target_count,
        "frequency": frequency.value,
        "observations": observations,
        "rebalance_count": len(rebalance_dates),
        "annual_return": annual_strategy,
        "annual_benchmark_return": annual_benchmark,
        "annual_excess_return": annual_excess,
        "excess_sharpe": excess_sharpe,
        "annual_turnover": total_turnover * 252 / observations,
        "maximum_drawdown": _maximum_drawdown(strategy),
        "selection_score": annual_excess + 0.05 * excess_sharpe,
    }


def _annualized(returns: FloatArray) -> float:
    total = float(np.prod(1 + returns))
    return total ** (252 / len(returns)) - 1 if total > 0 else -1.0


def _maximum_drawdown(returns: FloatArray) -> float:
    equity = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(equity)
    return float(np.min(equity / peaks - 1))
