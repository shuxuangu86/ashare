from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import numpy.typing as npt

FloatMatrix = npt.NDArray[np.float64]
BoolMatrix = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class RegimeUniversePanel:
    dates: tuple[date, ...]
    ts_codes: tuple[str, ...]
    memberships: Mapping[str, BoolMatrix]

    def __post_init__(self) -> None:
        expected = (len(self.dates), len(self.ts_codes))
        if any(matrix.shape != expected for matrix in self.memberships.values()):
            raise ValueError("regime universe membership shape mismatch")
        if any(left >= right for left, right in zip(self.dates, self.dates[1:], strict=False)):
            raise ValueError("regime universe dates must be strictly increasing")


def build_dynamic_universes(
    dates: Sequence[date],
    *,
    close: FloatMatrix,
    total_market_cap: FloatMatrix,
    pb: FloatMatrix,
    turnover_rate: FloatMatrix,
    dividend_yield: FloatMatrix,
    netprofit_yoy: FloatMatrix,
    microcap_count: int = 400,
    quantile_fraction: float = 0.3,
) -> Mapping[str, BoolMatrix]:
    """Build lagged monthly universes from fields visible at each preceding close."""
    shape = close.shape
    fields = (total_market_cap, pb, turnover_rate, dividend_yield, netprofit_yoy)
    if any(field.shape != shape for field in fields) or shape[0] != len(dates):
        raise ValueError("dynamic universe input shape mismatch")
    if microcap_count <= 0 or not 0.0 < quantile_fraction <= 0.5:
        raise ValueError("dynamic universe selection parameters are invalid")

    returns = np.full(shape, np.nan, dtype=close.dtype)
    valid_pair = (
        np.isfinite(close[1:]) & np.isfinite(close[:-1]) & (close[:-1] > 0) & (close[1:] > 0)
    )
    returns[1:][valid_pair] = close[1:][valid_pair] / close[:-1][valid_pair] - 1.0
    all_a = np.isfinite(close) & (close > 0) & np.isfinite(total_market_cap)
    direct_definitions: dict[str, tuple[FloatMatrix, bool, int | None]] = {
        "MICROCAP_FIXED_MONTHLY": (total_market_cap, False, microcap_count),
        "MICROCAP_DYNAMIC": (total_market_cap, False, microcap_count),
        "LARGECAP_DYNAMIC": (total_market_cap, True, None),
        "DIVIDEND_DYNAMIC": (dividend_yield, True, None),
        "LOW_PRICE_DYNAMIC": (close, False, None),
        "HIGH_PRICE_DYNAMIC": (close, True, None),
        "HIGH_TURNOVER_DYNAMIC": (turnover_rate, True, None),
        "LOW_TURNOVER_DYNAMIC": (turnover_rate, False, None),
    }
    output: dict[str, BoolMatrix] = {"ALL_A": all_a}
    for name, (score, high, count) in direct_definitions.items():
        monthly = name != "MICROCAP_DYNAMIC"
        output[name] = _lagged_selection(
            dates,
            score,
            all_a,
            high=high,
            count=count,
            fraction=quantile_fraction,
            monthly=monthly,
        )
    growth_score = _growth_score(netprofit_yoy, pb)
    output["GROWTH_DYNAMIC"] = _lagged_selection(
        dates,
        growth_score,
        all_a,
        high=True,
        count=None,
        fraction=quantile_fraction,
        monthly=True,
    )
    del growth_score
    value_score = _value_score(pb, dividend_yield)
    output["VALUE_DYNAMIC"] = _lagged_selection(
        dates,
        value_score,
        all_a,
        high=True,
        count=None,
        fraction=quantile_fraction,
        monthly=True,
    )
    del value_score
    volatility = _rolling_nanstd(returns, 20)
    output["HIGH_VOL_DYNAMIC"] = _lagged_selection(
        dates,
        volatility,
        all_a,
        high=True,
        count=None,
        fraction=quantile_fraction,
        monthly=True,
    )
    output["LOW_VOL_DYNAMIC"] = _lagged_selection(
        dates,
        volatility,
        all_a,
        high=False,
        count=None,
        fraction=quantile_fraction,
        monthly=True,
    )
    return output


def build_snapshot_membership(
    dates: Sequence[date],
    ts_codes: Sequence[str],
    snapshots: Mapping[date, Sequence[str]],
    *,
    minimum_constituents: int,
) -> BoolMatrix:
    """Forward-fill only complete historical snapshots after their effective date."""
    if minimum_constituents <= 0:
        raise ValueError("minimum_constituents must be positive")
    code_positions = {code: index for index, code in enumerate(ts_codes)}
    complete = sorted(
        (snapshot_date, tuple(dict.fromkeys(codes)))
        for snapshot_date, codes in snapshots.items()
        if len(set(codes)) >= minimum_constituents
    )
    output = np.zeros((len(dates), len(ts_codes)), dtype=np.bool_)
    snapshot_index = -1
    active: tuple[str, ...] = ()
    for date_index, trade_date in enumerate(dates):
        while snapshot_index + 1 < len(complete) and complete[snapshot_index + 1][0] < trade_date:
            snapshot_index += 1
            active = complete[snapshot_index][1]
        for code in active:
            position = code_positions.get(code)
            if position is not None:
                output[date_index, position] = True
    return output


def _lagged_selection(
    dates: Sequence[date],
    score: FloatMatrix,
    eligible: BoolMatrix,
    *,
    high: bool,
    count: int | None,
    fraction: float,
    monthly: bool,
) -> BoolMatrix:
    output = np.zeros(score.shape, dtype=np.bool_)
    active = np.zeros(score.shape[1], dtype=np.bool_)
    for date_index in range(1, len(dates)):
        rebalance = not monthly or (
            dates[date_index].year,
            dates[date_index].month,
        ) != (
            dates[date_index - 1].year,
            dates[date_index - 1].month,
        )
        if rebalance:
            visible = score[date_index - 1]
            candidates = np.flatnonzero(eligible[date_index - 1] & np.isfinite(visible))
            target = (
                min(count, len(candidates))
                if count is not None
                else max(1, int(len(candidates) * fraction))
            )
            ordered = candidates[np.argsort(visible[candidates], kind="stable")]
            chosen = ordered[-target:] if high else ordered[:target]
            active = np.zeros(score.shape[1], dtype=np.bool_)
            active[chosen] = True
        output[date_index] = active
    return output


def _rolling_nanstd(values: FloatMatrix, window: int) -> FloatMatrix:
    output = np.full(values.shape, np.nan, dtype=values.dtype)
    for index in range(window - 1, values.shape[0]):
        sample = values[index - window + 1 : index + 1]
        counts = np.sum(np.isfinite(sample), axis=0)
        valid = counts >= window // 2
        output[index, valid] = np.nanstd(sample[:, valid], axis=0)
    return output


def _row_percentile(values: FloatMatrix, *, ascending: bool = True) -> FloatMatrix:
    output = np.full(values.shape, np.nan, dtype=np.float32)
    for index, row in enumerate(values):
        valid = np.flatnonzero(np.isfinite(row))
        if not len(valid):
            continue
        ordered = valid[np.argsort(row[valid], kind="stable")]
        ranks = np.linspace(0.0, 1.0, len(ordered), endpoint=True)
        output[index, ordered] = ranks if ascending else ranks[::-1]
    return cast(FloatMatrix, output)


def _growth_score(netprofit_yoy: FloatMatrix, pb: FloatMatrix) -> FloatMatrix:
    return _row_percentile(netprofit_yoy) + _row_percentile(pb)


def _value_score(pb: FloatMatrix, dividend_yield: FloatMatrix) -> FloatMatrix:
    valid_pb = np.where(pb > 0, pb, np.nan)
    return _row_percentile(valid_pb, ascending=False) + _row_percentile(dividend_yield)
