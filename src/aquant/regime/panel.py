from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt

from aquant.regime.percentiles import expanding_percentile, rolling_percentile
from aquant.regime.universe import BoolMatrix, FloatMatrix

FloatSeries = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MarketStatePanelInput:
    dates: tuple[date, ...]
    ts_codes: tuple[str, ...]
    close: FloatMatrix
    amount: FloatMatrix
    total_market_cap: FloatMatrix
    float_market_cap: FloatMatrix
    pb: FloatMatrix
    turnover_rate: FloatMatrix
    dividend_yield: FloatMatrix
    netprofit_yoy: FloatMatrix
    debt_to_assets: FloatMatrix
    up_limit: FloatMatrix
    down_limit: FloatMatrix
    universes: Mapping[str, BoolMatrix]
    index_levels: Mapping[str, FloatSeries]

    def __post_init__(self) -> None:
        shape = (len(self.dates), len(self.ts_codes))
        matrices = (
            self.close,
            self.amount,
            self.total_market_cap,
            self.float_market_cap,
            self.pb,
            self.turnover_rate,
            self.dividend_yield,
            self.netprofit_yoy,
            self.debt_to_assets,
            self.up_limit,
            self.down_limit,
        )
        if any(matrix.shape != shape for matrix in matrices):
            raise ValueError("market-state panel matrix shape mismatch")
        if any(matrix.shape != shape for matrix in self.universes.values()):
            raise ValueError("market-state universe shape mismatch")
        if any(series.shape != (len(self.dates),) for series in self.index_levels.values()):
            raise ValueError("market-state index level shape mismatch")


def compute_market_state_panel(panel: MarketStatePanelInput) -> Mapping[str, FloatSeries]:
    """Compute every state supported by the current Standard/PIT release."""
    stock_returns = _stock_returns(panel.close)
    levels = _scope_levels(panel, stock_returns)
    output: dict[str, FloatSeries] = {}
    _compute_style(output, levels)
    _compute_trend(output, levels)
    _compute_breadth(output, panel, stock_returns)
    _compute_valuation(output, panel)
    _compute_liquidity(output, panel)
    _compute_crowding(output, panel, levels)
    _compute_risk_appetite(output, levels)
    _compute_fundamentals(output, panel)
    return output


def _scope_levels(
    panel: MarketStatePanelInput,
    stock_returns: FloatMatrix,
) -> dict[str, FloatSeries]:
    level_sources = {
        "hs300": "HS300",
        "csi500": "CSI500",
        "csi1000": "CSI1000",
        "chinext": "CHINEXT",
    }
    levels = {
        prefix: np.asarray(panel.index_levels[name], dtype=np.float64)
        for prefix, name in level_sources.items()
        if name in panel.index_levels
    }
    universe_sources = {
        "all_a": "ALL_A",
        "microcap": "MICROCAP_FIXED_MONTHLY",
        "growth": "GROWTH_DYNAMIC",
        "value": "VALUE_DYNAMIC",
        "dividend": "DIVIDEND_DYNAMIC",
        "largecap": "LARGECAP_DYNAMIC",
        "high_vol": "HIGH_VOL_DYNAMIC",
        "low_vol": "LOW_VOL_DYNAMIC",
        "low_price": "LOW_PRICE_DYNAMIC",
        "high_price": "HIGH_PRICE_DYNAMIC",
        "high_turnover": "HIGH_TURNOVER_DYNAMIC",
        "low_turnover": "LOW_TURNOVER_DYNAMIC",
    }
    for prefix, universe in universe_sources.items():
        if universe in panel.universes:
            levels[prefix] = _equal_weight_level(stock_returns, panel.universes[universe])
    return levels


def _compute_style(output: dict[str, FloatSeries], levels: Mapping[str, FloatSeries]) -> None:
    for style in ("growth", "value"):
        if style not in levels:
            continue
        for window in (5, 20, 40, 60, 120):
            output[f"{style}_return_{window}d"] = _returns(levels[style], window)
    if "growth" in levels and "value" in levels:
        for window in (5, 20, 40, 60, 120):
            output[f"growth_minus_value_return_{window}d"] = _relative_return(
                levels["growth"], levels["value"], window
            )


def _compute_trend(output: dict[str, FloatSeries], levels: Mapping[str, FloatSeries]) -> None:
    for prefix in ("all_a", "hs300", "csi500", "csi1000", "microcap", "chinext"):
        level = levels.get(prefix)
        if level is None:
            continue
        for window in (5, 20, 40, 60, 120):
            output[f"{prefix}_return_{window}d"] = _returns(level, window)
        for window in (20, 60, 120):
            output[f"{prefix}_drawdown_{window}d"] = _drawdown(level, window)
            output[f"{prefix}_distance_to_ma{window}"] = _distance_to_mean(level, window)
            output[f"{prefix}_realized_volatility_{window}d"] = _realized_volatility(level, window)
        for window in (20, 60):
            output[f"{prefix}_trend_slope_{window}d"] = _slope(level, window)


def _compute_breadth(
    output: dict[str, FloatSeries],
    panel: MarketStatePanelInput,
    stock_returns: FloatMatrix,
) -> None:
    scopes = {
        "all_a": "ALL_A",
        "hs300": "HS300",
        "csi500": "CSI500",
        "csi1000": "CSI1000",
        "microcap": "MICROCAP_FIXED_MONTHLY",
        "growth": "GROWTH_DYNAMIC",
        "value": "VALUE_DYNAMIC",
        "dividend": "DIVIDEND_DYNAMIC",
    }
    positive = stock_returns > 0
    negative = stock_returns < 0
    for prefix, universe in scopes.items():
        mask = panel.universes.get(universe)
        if mask is None:
            continue
        output[f"{prefix}_positive_return_ratio"] = _masked_ratio(positive, mask)
    for window in (5, 20, 60, 120):
        average = _rolling_matrix_mean(panel.close, window)
        for prefix, universe in scopes.items():
            mask = panel.universes.get(universe)
            if mask is None:
                continue
            output[f"{prefix}_above_ma{window}_ratio"] = _masked_ratio(
                panel.close > average, mask & np.isfinite(average)
            )
        del average
    for window in (60, 252):
        high = _rolling_matrix_extreme(panel.close, window, maximum=True)
        low = _rolling_matrix_extreme(panel.close, window, maximum=False)
        for prefix, universe in scopes.items():
            mask = panel.universes.get(universe)
            if mask is None:
                continue
            output[f"{prefix}_new_high_{window}d_ratio"] = _masked_ratio(
                panel.close >= high,
                mask & np.isfinite(high),
            )
            output[f"{prefix}_new_low_{window}d_ratio"] = _masked_ratio(
                panel.close <= low,
                mask & np.isfinite(low),
            )
        del high, low
    for prefix, universe in scopes.items():
        mask = panel.universes.get(universe)
        if mask is None:
            continue
        advances = np.sum(positive & mask, axis=1)
        declines = np.sum(negative & mask, axis=1)
        output[f"{prefix}_advance_decline_ratio"] = advances / np.maximum(declines, 1)
        output[f"{prefix}_limit_up_ratio"] = _masked_ratio(
            panel.close >= panel.up_limit,
            mask & np.isfinite(panel.up_limit),
        )
        output[f"{prefix}_limit_down_ratio"] = _masked_ratio(
            panel.close <= panel.down_limit,
            mask & np.isfinite(panel.down_limit),
        )
        output[f"{prefix}_large_gain_ratio"] = _masked_ratio(stock_returns >= 0.05, mask)
        output[f"{prefix}_large_loss_ratio"] = _masked_ratio(stock_returns <= -0.05, mask)


def _compute_valuation(output: dict[str, FloatSeries], panel: MarketStatePanelInput) -> None:
    scopes = {
        "all_a": "ALL_A",
        "hs300": "HS300",
        "csi500": "CSI500",
        "csi1000": "CSI1000",
        "microcap": "MICROCAP_FIXED_MONTHLY",
        "growth": "GROWTH_DYNAMIC",
        "value": "VALUE_DYNAMIC",
        "dividend": "DIVIDEND_DYNAMIC",
    }
    for prefix, universe in scopes.items():
        mask = panel.universes.get(universe)
        if mask is None:
            continue
        valid = mask & np.isfinite(panel.pb) & (panel.pb > 0)
        median = _row_stat(panel.pb, valid, "median")
        output[f"{prefix}_pb_median"] = median
        for suffix, operation in (
            ("pb_winsorized_mean", "winsorized_mean"),
            ("pb_25pct", "q25"),
            ("pb_75pct", "q75"),
            ("pb_dispersion", "std"),
        ):
            output[f"{prefix}_{suffix}"] = _row_stat(panel.pb, valid, operation)
        denominator = np.nansum(np.where(valid, panel.total_market_cap / panel.pb, np.nan), axis=1)
        numerator = np.nansum(np.where(valid, panel.total_market_cap, np.nan), axis=1)
        output[f"{prefix}_aggregate_pb"] = np.divide(
            numerator,
            denominator,
            out=np.full(len(panel.dates), np.nan),
            where=denominator > 0,
        )
        output[f"{prefix}_pb_percentile_3y"] = _rolling_percentile_array(median, 756, 252)
        output[f"{prefix}_pb_percentile_5y"] = _rolling_percentile_array(median, 1260, 252)
        output[f"{prefix}_pb_percentile_expanding"] = np.asarray(
            expanding_percentile(median.tolist(), minimum_periods=252), dtype=np.float64
        )
        output[f"{prefix}_pb_change_20d"] = _returns(median, 20)


def _compute_liquidity(output: dict[str, FloatSeries], panel: MarketStatePanelInput) -> None:
    all_a = panel.universes.get("ALL_A")
    hs300 = panel.universes.get("HS300")
    csi1000 = panel.universes.get("CSI1000")
    if all_a is not None and hs300 is not None:
        total = np.nansum(np.where(all_a, panel.amount, np.nan), axis=1)
        member = np.nansum(np.where(hs300, panel.amount, np.nan), axis=1)
        share = np.divide(member, total, out=np.full(len(panel.dates), np.nan), where=total > 0)
        output["hs300_amount_share_all_a"] = share
        output["hs300_amount_share_20d"] = _rolling_mean(share, 20)
        output["hs300_amount_share_percentile_3y"] = _rolling_percentile_array(share, 756, 252)
    if csi1000 is not None:
        amount = np.nansum(np.where(csi1000, panel.amount, np.nan), axis=1)
        float_mv = np.nansum(np.where(csi1000, panel.float_market_cap, np.nan), axis=1)
        turnover = np.divide(
            amount,
            float_mv,
            out=np.full(len(panel.dates), np.nan),
            where=float_mv > 0,
        )
        output["csi1000_turnover_daily"] = turnover
        output["csi1000_turnover_20d"] = _rolling_mean(turnover, 20)
        output["csi1000_turnover_percentile_3y"] = _rolling_percentile_array(turnover, 756, 252)


def _compute_crowding(
    output: dict[str, FloatSeries],
    panel: MarketStatePanelInput,
    levels: Mapping[str, FloatSeries],
) -> None:
    microcap = panel.universes.get("MICROCAP_FIXED_MONTHLY")
    all_a = panel.universes.get("ALL_A")
    if microcap is None or all_a is None or "microcap" not in levels or "all_a" not in levels:
        return
    amount_total = np.nansum(np.where(all_a, panel.amount, np.nan), axis=1)
    amount_micro = np.nansum(np.where(microcap, panel.amount, np.nan), axis=1)
    amount_share = np.divide(
        amount_micro,
        amount_total,
        out=np.full(len(panel.dates), np.nan),
        where=amount_total > 0,
    )
    float_mv = np.nansum(np.where(microcap, panel.float_market_cap, np.nan), axis=1)
    turnover = np.divide(
        amount_micro,
        float_mv,
        out=np.full(len(panel.dates), np.nan),
        where=float_mv > 0,
    )
    micro_pb = output.get("microcap_aggregate_pb")
    all_pb = output.get("all_a_aggregate_pb")
    relative_valuation = (
        np.divide(micro_pb, all_pb, out=np.full(len(panel.dates), np.nan), where=all_pb > 0)
        if micro_pb is not None and all_pb is not None
        else np.full(len(panel.dates), np.nan)
    )
    relative_momentum = _relative_return(levels["microcap"], levels["all_a"], 60)
    breadth = output.get("microcap_above_ma20_ratio", np.full(len(panel.dates), np.nan))
    raw = {
        "microcap_amount_share": amount_share,
        "microcap_turnover": turnover,
        "microcap_relative_valuation": relative_valuation,
        "microcap_relative_momentum": relative_momentum,
        "microcap_breadth": breadth,
    }
    normalized = {name: _rolling_percentile_array(values, 756, 252) for name, values in raw.items()}
    output.update(raw)
    stack = np.vstack(tuple(normalized.values()))
    coverage = np.sum(np.isfinite(stack), axis=0)
    score = np.divide(
        np.nansum(stack, axis=0),
        coverage,
        out=np.full(len(panel.dates), np.nan),
        where=coverage > 0,
    )
    score[coverage < 3] = np.nan
    output["microcap_crowding_score_v1"] = score


def _compute_risk_appetite(
    output: dict[str, FloatSeries], levels: Mapping[str, FloatSeries]
) -> None:
    pairs = {
        "microcap_minus_largecap": ("microcap", "largecap"),
        "high_vol_minus_low_vol": ("high_vol", "low_vol"),
        "low_price_minus_high_price": ("low_price", "high_price"),
        "high_turnover_minus_low_turnover": ("high_turnover", "low_turnover"),
        "growth_minus_dividend": ("growth", "dividend"),
    }
    for prefix, (left, right) in pairs.items():
        if left not in levels or right not in levels:
            continue
        for window in (20, 40, 60):
            output[f"{prefix}_return_{window}d"] = _relative_return(
                levels[left], levels[right], window
            )


def _compute_fundamentals(output: dict[str, FloatSeries], panel: MarketStatePanelInput) -> None:
    scopes = {
        "all_a": "ALL_A",
        "growth": "GROWTH_DYNAMIC",
        "value": "VALUE_DYNAMIC",
        "microcap": "MICROCAP_FIXED_MONTHLY",
        "dividend": "DIVIDEND_DYNAMIC",
    }
    for prefix, universe in scopes.items():
        mask = panel.universes.get(universe)
        if mask is None:
            continue
        profit_valid = mask & np.isfinite(panel.netprofit_yoy)
        debt_valid = mask & np.isfinite(panel.debt_to_assets)
        output[f"{prefix}_profit_growth_median"] = _row_stat(
            panel.netprofit_yoy, profit_valid, "median"
        )
        output[f"{prefix}_negative_profit_growth_ratio"] = _masked_ratio(
            panel.netprofit_yoy < 0, profit_valid
        )
        output[f"{prefix}_debt_to_assets_median"] = _row_stat(
            panel.debt_to_assets, debt_valid, "median"
        )


def _stock_returns(close: FloatMatrix) -> FloatMatrix:
    output = np.full(close.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(close[1:]) & np.isfinite(close[:-1]) & (close[1:] > 0) & (close[:-1] > 0)
    output[1:][valid] = close[1:][valid] / close[:-1][valid] - 1.0
    return output


def _equal_weight_level(returns: FloatMatrix, membership: BoolMatrix) -> FloatSeries:
    effective = np.zeros_like(membership)
    effective[1:] = membership[:-1]
    counts = np.sum(effective & np.isfinite(returns), axis=1)
    daily = np.divide(
        np.nansum(np.where(effective, returns, np.nan), axis=1),
        counts,
        out=np.zeros(returns.shape[0], dtype=np.float64),
        where=counts > 0,
    )
    return np.cumprod(1.0 + daily)


def _returns(values: FloatSeries, window: int) -> FloatSeries:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values[window:]) & np.isfinite(values[:-window]) & (values[:-window] > 0)
    output[window:][valid] = values[window:][valid] / values[:-window][valid] - 1.0
    return output


def _relative_return(left: FloatSeries, right: FloatSeries, window: int) -> FloatSeries:
    left_return = _returns(left, window)
    right_return = _returns(right, window)
    result: FloatSeries = np.divide(
        1.0 + left_return,
        1.0 + right_return,
        out=np.full(left.shape, np.nan),
        where=np.isfinite(left_return) & np.isfinite(right_return),
    )
    return result - 1.0


def _rolling_mean(values: FloatSeries, window: int) -> FloatSeries:
    output = np.full(values.shape, np.nan)
    for index in range(window - 1, len(values)):
        sample = values[index - window + 1 : index + 1]
        if np.all(np.isfinite(sample)):
            output[index] = np.mean(sample)
    return output


def _drawdown(values: FloatSeries, window: int) -> FloatSeries:
    output = np.full(values.shape, np.nan)
    for index in range(window - 1, len(values)):
        sample = values[index - window + 1 : index + 1]
        if np.all(np.isfinite(sample)):
            output[index] = values[index] / np.max(sample) - 1.0
    return output


def _distance_to_mean(values: FloatSeries, window: int) -> FloatSeries:
    average = _rolling_mean(values, window)
    result: FloatSeries = np.divide(
        values,
        average,
        out=np.full(values.shape, np.nan),
        where=np.isfinite(average) & (average != 0),
    )
    return result - 1.0


def _realized_volatility(values: FloatSeries, window: int) -> FloatSeries:
    returns = np.full(values.shape, np.nan)
    valid = (
        np.isfinite(values[1:]) & np.isfinite(values[:-1]) & (values[1:] > 0) & (values[:-1] > 0)
    )
    returns[1:][valid] = np.log(values[1:][valid] / values[:-1][valid])
    output = np.full(values.shape, np.nan)
    for index in range(window, len(values)):
        sample = returns[index - window + 1 : index + 1]
        if np.all(np.isfinite(sample)):
            output[index] = np.std(sample) * math.sqrt(252)
    return output


def _slope(values: FloatSeries, window: int) -> FloatSeries:
    output = np.full(values.shape, np.nan)
    x = np.arange(window, dtype=np.float64)
    centered = x - np.mean(x)
    denominator = np.sum(centered**2)
    for index in range(window - 1, len(values)):
        sample = values[index - window + 1 : index + 1]
        if np.all(np.isfinite(sample)) and np.all(sample > 0):
            output[index] = np.sum(centered * np.log(sample)) / denominator
    return output


def _rolling_matrix_mean(values: FloatMatrix, window: int) -> FloatMatrix:
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    sums = np.cumsum(filled, axis=0)
    counts = np.cumsum(valid, axis=0)
    output = np.full(values.shape, np.nan, dtype=values.dtype)
    for index in range(window - 1, values.shape[0]):
        prior_sums = sums[index - window] if index >= window else 0.0
        prior_counts = counts[index - window] if index >= window else 0
        window_counts = counts[index] - prior_counts
        window_sums = sums[index] - prior_sums
        complete = window_counts == window
        output[index, complete] = window_sums[complete] / window
    return output


def _rolling_matrix_extreme(values: FloatMatrix, window: int, *, maximum: bool) -> FloatMatrix:
    output = np.full(values.shape, np.nan, dtype=values.dtype)
    operation = np.nanmax if maximum else np.nanmin
    for index in range(window - 1, values.shape[0]):
        sample = values[index - window + 1 : index + 1]
        complete = np.sum(np.isfinite(sample), axis=0) == window
        if np.any(complete):
            output[index, complete] = operation(sample[:, complete], axis=0)
    return output


def _masked_ratio(condition: BoolMatrix, mask: BoolMatrix) -> FloatSeries:
    denominator = np.sum(mask, axis=1)
    result: FloatSeries = np.divide(
        np.sum(condition & mask, axis=1),
        denominator,
        out=np.full(mask.shape[0], np.nan),
        where=denominator > 0,
    )
    return result


def _row_stat(values: FloatMatrix, mask: BoolMatrix, operation: str) -> FloatSeries:
    output = np.full(values.shape[0], np.nan)
    for index in range(values.shape[0]):
        sample = values[index, mask[index] & np.isfinite(values[index])]
        if not len(sample):
            continue
        if operation == "median":
            output[index] = np.median(sample)
        elif operation == "winsorized_mean":
            lower, upper = np.quantile(sample, (0.01, 0.99))
            output[index] = np.mean(np.clip(sample, lower, upper))
        elif operation == "q25":
            output[index] = np.quantile(sample, 0.25)
        elif operation == "q75":
            output[index] = np.quantile(sample, 0.75)
        elif operation == "std":
            output[index] = np.std(sample)
        else:
            raise ValueError(f"unknown row statistic: {operation}")
    return output


def _rolling_percentile_array(
    values: FloatSeries, window: int, minimum_periods: int
) -> FloatSeries:
    return np.asarray(
        rolling_percentile(values.tolist(), window, minimum_periods=minimum_periods),
        dtype=np.float64,
    )
