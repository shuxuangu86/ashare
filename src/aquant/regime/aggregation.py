from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from aquant.regime.protocol import Numeric, StateSeries


def _valid(values: Sequence[Numeric]) -> list[float]:
    return [
        resolved
        for value in values
        if value is not None and math.isfinite(resolved := float(value))
    ]


def rolling_mean(
    values: Sequence[Numeric],
    window: int,
    *,
    minimum_periods: int | None = None,
) -> StateSeries:
    if window <= 0:
        raise ValueError("window must be positive")
    required = window if minimum_periods is None else minimum_periods
    if not 1 <= required <= window:
        raise ValueError("minimum_periods must be between one and window")
    output: list[float | None] = []
    for index in range(len(values)):
        sample = _valid(values[max(0, index - window + 1) : index + 1])
        output.append(sum(sample) / len(sample) if len(sample) >= required else None)
    return tuple(output)


def returns(values: Sequence[Numeric], window: int) -> StateSeries:
    if window <= 0:
        raise ValueError("window must be positive")
    output: list[float | None] = [None] * len(values)
    for index in range(window, len(values)):
        current, previous = values[index], values[index - window]
        if current is None or previous is None:
            continue
        current_value, previous_value = float(current), float(previous)
        if math.isfinite(current_value) and math.isfinite(previous_value) and previous_value > 0:
            output[index] = current_value / previous_value - 1.0
    return tuple(output)


def relative_returns(
    left_values: Sequence[Numeric],
    right_values: Sequence[Numeric],
    window: int,
) -> StateSeries:
    if len(left_values) != len(right_values):
        raise ValueError("relative-return inputs must have equal length")
    left = returns(left_values, window)
    right = returns(right_values, window)
    return tuple(
        None
        if left_value is None or right_value is None
        else (1.0 + left_value) / (1.0 + right_value) - 1.0
        for left_value, right_value in zip(left, right, strict=True)
    )


def amount_share(
    member_amounts: Sequence[Numeric], all_a_amounts: Sequence[Numeric]
) -> StateSeries:
    if len(member_amounts) != len(all_a_amounts):
        raise ValueError("amount-share inputs must have equal length")
    output: list[float | None] = []
    for numerator, denominator in zip(member_amounts, all_a_amounts, strict=True):
        if numerator is None or denominator is None or float(denominator) <= 0:
            output.append(None)
            continue
        share = float(numerator) / float(denominator)
        output.append(share if 0.0 <= share <= 1.0 else None)
    return tuple(output)


def weighted_mean(values: Sequence[Numeric], weights: Sequence[Numeric]) -> float | None:
    if len(values) != len(weights):
        raise ValueError("values and weights must have equal length")
    pairs = [
        (float(value), float(weight))
        for value, weight in zip(values, weights, strict=True)
        if value is not None
        and weight is not None
        and math.isfinite(float(value))
        and math.isfinite(float(weight))
        and float(weight) > 0
    ]
    total_weight = sum(weight for _, weight in pairs)
    return (
        sum(value * weight for value, weight in pairs) / total_weight if total_weight > 0 else None
    )


def median(values: Sequence[Numeric], *, positive_only: bool = False) -> float | None:
    sample = _valid(values)
    if positive_only:
        sample = [value for value in sample if value > 0]
    return statistics.median(sample) if sample else None


def aggregate_turnover(
    amounts: Sequence[Numeric], free_float_market_values: Sequence[Numeric]
) -> float | None:
    numerator = sum(_valid(amounts))
    denominator = sum(value for value in _valid(free_float_market_values) if value > 0)
    return numerator / denominator if denominator > 0 else None
