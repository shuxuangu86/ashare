from __future__ import annotations

import math
from collections.abc import Sequence

from aquant.regime.protocol import Numeric, StateSeries


def _finite(value: Numeric) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def percentile_rank(history: Sequence[Numeric], current: Numeric) -> float | None:
    """Return an average-tie empirical CDF rank in [0, 1]."""
    resolved = _finite(current)
    sample = [value for item in history if (value := _finite(item)) is not None]
    if resolved is None or not sample:
        return None
    below = sum(value < resolved for value in sample)
    equal = sum(value == resolved for value in sample)
    return (below + 0.5 * equal) / len(sample)


def rolling_percentile(
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
    for index, value in enumerate(values):
        history = values[max(0, index - window + 1) : index + 1]
        observed = sum(_finite(item) is not None for item in history)
        output.append(percentile_rank(history, value) if observed >= required else None)
    return tuple(output)


def expanding_percentile(
    values: Sequence[Numeric],
    *,
    minimum_periods: int = 1,
) -> StateSeries:
    if minimum_periods <= 0:
        raise ValueError("minimum_periods must be positive")
    output: list[float | None] = []
    observed = 0
    for index, value in enumerate(values):
        if _finite(value) is not None:
            observed += 1
        output.append(
            percentile_rank(values[: index + 1], value) if observed >= minimum_periods else None
        )
    return tuple(output)
