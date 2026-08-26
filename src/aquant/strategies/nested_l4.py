from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import numpy.typing as npt

from aquant.data.history.microcap import HistoryReleaseReader
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
    target_counts: tuple[int, ...] = (20, 30, 50, 80, 100),
    frequencies: tuple[RebalanceFrequency, ...] = (
        RebalanceFrequency.WEEKLY,
        RebalanceFrequency.MONTHLY,
    ),
    cost_bps: float = 10.0,
    benchmark_returns: npt.ArrayLike | None = None,
    eligibility_mask: npt.ArrayLike | None = None,
    open_prices: npt.ArrayLike | None = None,
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
    benchmark = (
        None if benchmark_returns is None else np.asarray(benchmark_returns, dtype=np.float64)
    )
    eligibility = None if eligibility_mask is None else np.asarray(eligibility_mask, dtype=bool)
    opening = None if open_prices is None else np.asarray(open_prices, dtype=np.float64)
    if benchmark is not None and benchmark.shape != (len(prices),):
        raise ValueError("L4 benchmark returns must align with trade dates")
    if eligibility is not None and eligibility.shape != prices.shape:
        raise ValueError("L4 eligibility mask must align with scores")
    if opening is not None and opening.shape != prices.shape:
        raise ValueError("L4 open prices must align with close")
    candidates = [
        _simulate(
            scores=values,
            close=prices,
            trade_dates=trade_dates,
            positions=validation_positions,
            target_count=target_count,
            frequency=frequency,
            cost_bps=cost_bps,
            benchmark_returns=benchmark,
            eligibility_mask=eligibility,
            open_prices=opening,
        )
        for target_count in target_counts
        for frequency in frequencies
    ]
    selected = max(
        candidates,
        key=lambda item: (
            bool(item["target_met"]),
            float(item["selection_score"]),
            -float(item["annual_turnover"]),
            -int(item["target_count"]),
        ),
    )
    return {
        "selection_window_start": trade_dates[int(validation_positions[0])].isoformat(),
        "selection_window_end": trade_dates[int(validation_positions[-1])].isoformat(),
        "cost_bps": cost_bps,
        "trial_count": len(candidates),
        "outer_test_used_for_selection": False,
        "objective": "annual_excess_return_plus_0.05_times_excess_sharpe",
        "target_definition": (
            "annual_excess_return>=0.15 OR (annual_excess_return>=0.10 AND excess_sharpe>0.8)"
        ),
        "execution_model": (
            "T_CLOSE_SIGNAL_T_PLUS_1_OPEN_WEIGHT_TRANSITION"
            if opening is not None
            else "T_CLOSE_SIGNAL_DELAYED_CLOSE_ACTIVATION_CONSERVATIVE_PROXY"
        ),
        "benchmark_model": (
            "PIT_ALL_A_SHARE_DAILY_EQUAL_PROXY"
            if benchmark is not None
            else "AVAILABLE_SECURITY_DAILY_EQUAL_CLOSE_RETURN_PROXY"
        ),
        "eligibility_model": (
            "PIT_LISTED_120D_NON_ST_NON_DELISTING_RISK"
            if eligibility is not None
            else "FINITE_SCORE_AND_CLOSE_ONLY"
        ),
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
    benchmark_returns: FloatArray | None,
    eligibility_mask: npt.NDArray[np.bool_] | None,
    open_prices: FloatArray | None,
) -> dict[str, Any]:
    dates = tuple(trade_dates[int(position)] for position in positions)
    rebalance_dates = generate_rebalance_dates(dates, frequency)
    active: FloatArray = np.zeros(close.shape[1], dtype=np.float64)
    pending: FloatArray | None = None
    strategy_returns: list[float] = []
    benchmark_path: list[float] = []
    total_turnover = 0.0
    for position in positions:
        index = int(position)
        day_turnover = 0.0
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
        if open_prices is not None:
            overnight_asset = np.divide(
                open_prices[index],
                close[index - 1],
                out=np.full(close.shape[1], np.nan),
                where=np.isfinite(open_prices[index])
                & np.isfinite(close[index - 1])
                & (close[index - 1] > 0),
            )
            overnight_asset -= 1
            valid_overnight = np.isfinite(overnight_asset)
            overnight_return = float(
                np.sum(active[valid_overnight] * overnight_asset[valid_overnight])
            )
            drifted = active.copy()
            if 1 + overnight_return > 0:
                drifted[valid_overnight] *= 1 + overnight_asset[valid_overnight]
                drifted /= 1 + overnight_return
            active = drifted
            if pending is not None:
                day_turnover = float(np.sum(np.abs(pending - active)))
                total_turnover += day_turnover
                active = pending
                pending = None
            intraday_asset = np.divide(
                close[index],
                open_prices[index],
                out=np.full(close.shape[1], np.nan),
                where=np.isfinite(close[index])
                & np.isfinite(open_prices[index])
                & (open_prices[index] > 0),
            )
            intraday_asset -= 1
            valid_intraday = np.isfinite(intraday_asset)
            intraday_return = float(np.sum(active[valid_intraday] * intraday_asset[valid_intraday]))
            portfolio_return = (
                (1 + overnight_return) * (1 + intraday_return)
                - 1
                - day_turnover * cost_bps / 10_000
            )
            close_drifted = active.copy()
            if 1 + intraday_return > 0:
                close_drifted[valid_intraday] *= 1 + intraday_asset[valid_intraday]
                close_drifted /= 1 + intraday_return
            active = close_drifted
        benchmark_return = (
            float(benchmark_returns[index])
            if benchmark_returns is not None and np.isfinite(benchmark_returns[index])
            else float(np.mean(asset_returns[valid_returns]))
            if np.any(valid_returns)
            else 0.0
        )
        if open_prices is None and pending is not None:
            turnover = float(np.sum(np.abs(pending - active)))
            portfolio_return -= turnover * cost_bps / 10_000
            total_turnover += turnover
            active[:] = pending
            pending = None
        strategy_returns.append(portfolio_return)
        benchmark_path.append(benchmark_return)
        if trade_dates[index] in rebalance_dates:
            row = scores[index]
            eligible_values = np.isfinite(row) & np.isfinite(close[index])
            if eligibility_mask is not None:
                eligible_values &= eligibility_mask[index]
            eligible = np.flatnonzero(eligible_values)
            selected = eligible[np.argsort(row[eligible], kind="stable")[-target_count:]]
            pending = np.zeros(close.shape[1], dtype=np.float64)
            if len(selected):
                pending[selected] = 1 / len(selected)
    strategy = np.asarray(strategy_returns)
    benchmark = np.asarray(benchmark_path)
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
    target_met = bool(annual_excess >= 0.15 or (annual_excess >= 0.10 and excess_sharpe > 0.8))
    return {
        "target_count": target_count,
        "frequency": frequency.value,
        "observations": observations,
        "rebalance_count": len(rebalance_dates),
        "annual_return": annual_strategy,
        "annual_benchmark_return": annual_benchmark,
        "annual_excess_return": annual_excess,
        "excess_sharpe": excess_sharpe,
        "target_met": target_met,
        "annual_turnover": total_turnover * 252 / observations,
        "maximum_drawdown": _maximum_drawdown(strategy),
        "selection_score": annual_excess + 0.05 * excess_sharpe,
    }


def build_l4_eligibility_matrix(
    release_directory: Path,
    *,
    trade_dates: tuple[date, ...],
    ts_codes: tuple[str, ...],
    minimum_listing_days: int = 120,
) -> npt.NDArray[np.bool_]:
    """Build the close-time PIT universe used by both inner selection and L4."""
    if not trade_dates or not ts_codes or minimum_listing_days < 0:
        raise ValueError("L4 eligibility inputs are invalid")
    release = HistoryReleaseReader(release_directory)
    release.verify("daily", "stock_basic", "namechange")
    date_index = {value: index for index, value in enumerate(trade_dates)}
    code_index = {value: index for index, value in enumerate(ts_codes)}
    result = np.zeros((len(trade_dates), len(ts_codes)), dtype=bool)
    connection = duckdb.connect(":memory:")
    try:
        batches = connection.execute(
            """
            SELECT
                daily.trade_date,
                daily.ts_code,
                stock.list_date,
                stock.delist_date,
                historical.name
            FROM read_parquet(?) AS daily
            JOIN read_parquet(?) AS stock USING (ts_code)
            LEFT JOIN read_parquet(?) AS historical
              ON historical.ts_code = daily.ts_code
             AND historical.start_date <= daily.trade_date
             AND (historical.end_date IS NULL OR historical.end_date >= daily.trade_date)
            WHERE daily.trade_date BETWEEN ? AND ?
              AND daily.exchange IN ('XSHG', 'XSHE')
            QUALIFY row_number() OVER (
                PARTITION BY daily.trade_date, daily.ts_code
                ORDER BY historical.start_date DESC NULLS LAST,
                         historical.end_date DESC NULLS LAST
            ) = 1
            ORDER BY daily.trade_date, daily.ts_code
            """,
            [
                release.parquet_pattern("daily"),
                release.parquet_pattern("stock_basic"),
                release.parquet_pattern("namechange"),
                trade_dates[0],
                trade_dates[-1],
            ],
        ).fetch_record_batch(rows_per_batch=100_000)
        for batch in batches:
            for trade_date, ts_code, list_date, delist_date, name in zip(
                *(batch.column(index).to_pylist() for index in range(5)), strict=True
            ):
                row = date_index.get(trade_date)
                column = code_index.get(ts_code)
                if row is None or column is None or not isinstance(name, str):
                    continue
                upper_name = name.upper()
                result[row, column] = bool(
                    (trade_date - list_date).days >= minimum_listing_days
                    and (delist_date is None or delist_date >= trade_date)
                    and "ST" not in upper_name
                    and "退" not in name
                )
    finally:
        connection.close()
    return result


def _annualized(returns: FloatArray) -> float:
    total = float(np.prod(1 + returns))
    return total ** (252 / len(returns)) - 1 if total > 0 else -1.0


def _maximum_drawdown(returns: FloatArray) -> float:
    equity = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(equity)
    return float(np.min(equity / peaks - 1))
