from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt

from aquant.factors.operators.math import safe_div

Array = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
NullPolicy = Literal["omit", "propagate"]
Alignment = Literal["trailing"]


def _matrix(values: npt.ArrayLike) -> tuple[Array, bool]:
    resolved = np.asarray(values, dtype=np.float64)
    if resolved.ndim not in {1, 2}:
        raise ValueError("time-series operators require a 1D or 2D array")
    return (resolved[:, None], True) if resolved.ndim == 1 else (resolved, False)


def _restore(values: Array, squeezed: bool) -> Array:
    return values[:, 0] if squeezed else values


def _validate(
    window: int,
    min_periods: int | None,
    null_policy: NullPolicy,
    alignment: Alignment,
) -> int:
    if window <= 0:
        raise ValueError("window must be positive")
    minimum = window if min_periods is None else min_periods
    if minimum <= 0 or minimum > window:
        raise ValueError("min_periods must be in [1, window]")
    if null_policy not in {"omit", "propagate"}:
        raise ValueError("null_policy must be omit or propagate")
    if alignment != "trailing":
        raise ValueError("only trailing alignment is permitted")
    return minimum


def delay(values: npt.ArrayLike, periods: int = 1) -> Array:
    if periods < 0:
        raise ValueError("delay periods must be non-negative")
    matrix, squeezed = _matrix(values)
    result = np.full(matrix.shape, np.nan)
    if periods == 0:
        result[:] = matrix
    elif periods < len(matrix):
        result[periods:] = matrix[:-periods]
    return _restore(result, squeezed)


def delta(values: npt.ArrayLike, periods: int = 1) -> Array:
    matrix, squeezed = _matrix(values)
    result = matrix - _matrix(delay(matrix, periods))[0]
    result[~np.isfinite(result)] = np.nan
    return _restore(result, squeezed)


def returns(values: npt.ArrayLike, periods: int = 1) -> Array:
    matrix, squeezed = _matrix(values)
    lagged = _matrix(delay(matrix, periods))[0]
    result = safe_div(matrix, lagged) - 1.0
    return _restore(result, squeezed)


def _rolling(
    values: npt.ArrayLike,
    *,
    window: int,
    min_periods: int | None,
    null_policy: NullPolicy,
    alignment: Alignment,
    reducer: Callable[[Array], float],
) -> Array:
    minimum = _validate(window, min_periods, null_policy, alignment)
    matrix, squeezed = _matrix(values)
    result = np.full(matrix.shape, np.nan)
    for row in range(len(matrix)):
        start = max(0, row - window + 1)
        for column in range(matrix.shape[1]):
            sample = matrix[start : row + 1, column]
            valid = sample[np.isfinite(sample)]
            if len(valid) < minimum or (null_policy == "propagate" and len(valid) != len(sample)):
                continue
            result[row, column] = reducer(valid)
    result[~np.isfinite(result)] = np.nan
    return _restore(result, squeezed)


def _rolling_moments(
    values: npt.ArrayLike,
    *,
    window: int,
    min_periods: int | None,
    null_policy: NullPolicy,
    alignment: Alignment,
) -> tuple[Array, Array, Array, BoolArray, bool]:
    minimum = _validate(window, min_periods, null_policy, alignment)
    matrix, squeezed = _matrix(values)
    finite = np.isfinite(matrix)
    filled = np.where(finite, matrix, 0)
    counts = np.vstack(
        (
            np.zeros((1, matrix.shape[1]), dtype=np.int64),
            np.cumsum(finite, axis=0, dtype=np.int64),
        )
    )
    sums = np.vstack((np.zeros((1, matrix.shape[1])), np.cumsum(filled, axis=0)))
    squares = np.vstack((np.zeros((1, matrix.shape[1])), np.cumsum(filled * filled, axis=0)))
    ends = np.arange(1, len(matrix) + 1)
    starts = np.maximum(ends - window, 0)
    rolling_count = counts[ends] - counts[starts]
    rolling_sum = sums[ends] - sums[starts]
    rolling_squares = squares[ends] - squares[starts]
    lengths = (ends - starts)[:, None]
    valid = rolling_count >= minimum
    if null_policy == "propagate":
        valid &= rolling_count == lengths
    return (
        rolling_count.astype(np.float64),
        rolling_sum,
        rolling_squares,
        valid.astype(np.bool_),
        squeezed,
    )


def rolling_sum(
    values: npt.ArrayLike,
    window: int,
    *,
    min_periods: int | None = None,
    null_policy: NullPolicy = "omit",
    alignment: Alignment = "trailing",
) -> Array:
    _counts, sums, _squares, valid, squeezed = _rolling_moments(
        values,
        window=window,
        min_periods=min_periods,
        null_policy=null_policy,
        alignment=alignment,
    )
    result = np.where(valid, sums, np.nan)
    return _restore(result, squeezed)


def rolling_mean(
    values: npt.ArrayLike,
    window: int,
    *,
    min_periods: int | None = None,
    null_policy: NullPolicy = "omit",
    alignment: Alignment = "trailing",
) -> Array:
    counts, sums, _squares, valid, squeezed = _rolling_moments(
        values,
        window=window,
        min_periods=min_periods,
        null_policy=null_policy,
        alignment=alignment,
    )
    with np.errstate(all="ignore"):
        result = np.where(valid, sums / counts, np.nan)
    return _restore(result, squeezed)


def rolling_var(
    values: npt.ArrayLike,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 1,
    null_policy: NullPolicy = "omit",
    alignment: Alignment = "trailing",
) -> Array:
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    counts, sums, squares, valid, squeezed = _rolling_moments(
        values,
        window=window,
        min_periods=min_periods,
        null_policy=null_policy,
        alignment=alignment,
    )
    valid &= counts > ddof
    with np.errstate(all="ignore"):
        numerator = squares - sums * sums / counts
        result = np.where(valid, np.maximum(numerator, 0) / (counts - ddof), np.nan)
    return _restore(result, squeezed)


def rolling_std(
    values: npt.ArrayLike,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 1,
    null_policy: NullPolicy = "omit",
    alignment: Alignment = "trailing",
) -> Array:
    return np.sqrt(
        rolling_var(
            values,
            window,
            min_periods=min_periods,
            ddof=ddof,
            null_policy=null_policy,
            alignment=alignment,
        )
    )


def _simple_reducer(operation: Callable[[Array], float]) -> Callable[..., Array]:
    def apply(
        values: npt.ArrayLike,
        window: int,
        *,
        min_periods: int | None = None,
        null_policy: NullPolicy = "omit",
        alignment: Alignment = "trailing",
    ) -> Array:
        return _rolling(
            values,
            window=window,
            min_periods=min_periods,
            null_policy=null_policy,
            alignment=alignment,
            reducer=operation,
        )

    return apply


rolling_min = _simple_reducer(lambda item: float(np.min(item)))
rolling_max = _simple_reducer(lambda item: float(np.max(item)))
rolling_median = _simple_reducer(lambda item: float(np.median(item)))


def rolling_quantile(
    values: npt.ArrayLike,
    window: int,
    quantile: float,
    *,
    min_periods: int | None = None,
    null_policy: NullPolicy = "omit",
    alignment: Alignment = "trailing",
) -> Array:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    return _rolling(
        values,
        window=window,
        min_periods=min_periods,
        null_policy=null_policy,
        alignment=alignment,
        reducer=lambda item: float(np.quantile(item, quantile)),
    )


def rolling_rank(
    values: npt.ArrayLike,
    window: int,
    *,
    min_periods: int | None = None,
    null_policy: NullPolicy = "omit",
    alignment: Alignment = "trailing",
) -> Array:
    return _rolling(
        values,
        window=window,
        min_periods=min_periods,
        null_policy=null_policy,
        alignment=alignment,
        reducer=lambda item: float(
            (np.count_nonzero(item < item[-1]) + 0.5 * np.count_nonzero(item == item[-1]))
            / len(item)
        ),
    )


def _rolling_pair(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
    window: int,
    *,
    min_periods: int | None,
    ddof: int,
    null_policy: NullPolicy,
    alignment: Alignment,
    covariance: bool,
) -> Array:
    minimum = _validate(window, min_periods, null_policy, alignment)
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    x, squeezed = _matrix(left)
    y, _ = _matrix(right)
    if x.shape != y.shape:
        raise ValueError("paired rolling inputs must have equal shape")
    finite = np.isfinite(x) & np.isfinite(y)
    paired_x = np.where(finite, x, 0)
    paired_y = np.where(finite, y, 0)

    def trailing_sum(values: Array) -> Array:
        cumulative = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(values, axis=0)))
        ends = np.arange(1, len(x) + 1)
        starts = np.maximum(ends - window, 0)
        return np.asarray(cumulative[ends] - cumulative[starts], dtype=np.float64)

    counts = trailing_sum(finite.astype(float))
    sum_x, sum_y = trailing_sum(paired_x), trailing_sum(paired_y)
    with np.errstate(all="ignore"):
        centered_x = trailing_sum(paired_x * paired_x) - sum_x * sum_x / counts
        centered_y = trailing_sum(paired_y * paired_y) - sum_y * sum_y / counts
        centered_xy = trailing_sum(paired_x * paired_y) - sum_x * sum_y / counts
    valid = (counts >= minimum) & (counts > ddof)
    if null_policy == "propagate":
        ends = np.arange(1, len(x) + 1)
        valid &= counts == (ends - np.maximum(ends - window, 0))[:, None]
    with np.errstate(all="ignore"):
        result = (
            centered_xy / (counts - ddof)
            if covariance
            else centered_xy / np.sqrt(centered_x * centered_y)
        )
    valid &= np.isfinite(result)
    result = np.where(valid, result, np.nan)
    return _restore(result, squeezed)


def rolling_corr(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 1,
    null_policy: NullPolicy = "omit",
    alignment: Alignment = "trailing",
) -> Array:
    return _rolling_pair(
        left,
        right,
        window,
        min_periods=min_periods,
        ddof=ddof,
        null_policy=null_policy,
        alignment=alignment,
        covariance=False,
    )


def rolling_cov(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 1,
    null_policy: NullPolicy = "omit",
    alignment: Alignment = "trailing",
) -> Array:
    return _rolling_pair(
        left,
        right,
        window,
        min_periods=min_periods,
        ddof=ddof,
        null_policy=null_policy,
        alignment=alignment,
        covariance=True,
    )


def rolling_skew(values: npt.ArrayLike, window: int, **kwargs: object) -> Array:
    return _rolling_standardized_moment(
        values,
        window=window,
        min_periods=kwargs.get("min_periods"),  # type: ignore[arg-type]
        null_policy=kwargs.get("null_policy", "omit"),  # type: ignore[arg-type]
        alignment=kwargs.get("alignment", "trailing"),  # type: ignore[arg-type]
        power=3,
    )


def rolling_kurt(values: npt.ArrayLike, window: int, **kwargs: object) -> Array:
    return _rolling_standardized_moment(
        values,
        window=window,
        min_periods=kwargs.get("min_periods"),  # type: ignore[arg-type]
        null_policy=kwargs.get("null_policy", "omit"),  # type: ignore[arg-type]
        alignment=kwargs.get("alignment", "trailing"),  # type: ignore[arg-type]
        power=4,
    )


def _rolling_standardized_moment(
    values: npt.ArrayLike,
    *,
    window: int,
    min_periods: int | None,
    null_policy: NullPolicy,
    alignment: Alignment,
    power: int,
) -> Array:
    counts, sums, squares, valid, squeezed = _rolling_moments(
        values,
        window=window,
        min_periods=min_periods,
        null_policy=null_policy,
        alignment=alignment,
    )
    matrix, _ = _matrix(values)
    filled = np.where(np.isfinite(matrix), matrix, 0)
    ends = np.arange(1, len(matrix) + 1)
    starts = np.maximum(ends - window, 0)

    def trailing_sum(samples: Array) -> Array:
        cumulative = np.vstack((np.zeros((1, matrix.shape[1])), np.cumsum(samples, axis=0)))
        return np.asarray(cumulative[ends] - cumulative[starts], dtype=np.float64)

    cubes = trailing_sum(filled**3)
    with np.errstate(all="ignore"):
        mean = sums / counts
        variance = squares / counts - mean * mean
        if power == 3:
            centered = cubes / counts - 3 * mean * squares / counts + 2 * mean**3
            result = centered / np.power(variance, 1.5)
        elif power == 4:
            fourth = trailing_sum(filled**4)
            centered = (
                fourth / counts
                - 4 * mean * cubes / counts
                + 6 * mean * mean * squares / counts
                - 3 * mean**4
            )
            result = centered / (variance * variance) - 3
        else:
            raise ValueError("standardized rolling moment supports power 3 or 4")
    valid &= variance > 0
    return _restore(np.where(valid, result, np.nan), squeezed)


def ts_zscore(values: npt.ArrayLike, window: int, *, min_periods: int | None = None) -> Array:
    mean = rolling_mean(values, window, min_periods=min_periods)
    std = rolling_std(values, window, min_periods=min_periods, ddof=0)
    return safe_div(np.asarray(values, dtype=np.float64) - mean, std)


def ema(values: npt.ArrayLike, span: int, *, min_periods: int = 1) -> Array:
    if span <= 0 or min_periods <= 0:
        raise ValueError("EMA span and min_periods must be positive")
    matrix, squeezed = _matrix(values)
    result = np.full(matrix.shape, np.nan)
    alpha = 2.0 / (span + 1)
    for column in range(matrix.shape[1]):
        state = np.nan
        count = 0
        for row, value in enumerate(matrix[:, column]):
            if not np.isfinite(value):
                continue
            count += 1
            state = value if not np.isfinite(state) else alpha * value + (1 - alpha) * state
            if count >= min_periods:
                result[row, column] = state
    return _restore(result, squeezed)


def wma(values: npt.ArrayLike, window: int, *, min_periods: int | None = None) -> Array:
    """Trailing linearly weighted moving average, newest observation has weight ``window``."""
    return _rolling(
        values,
        window=window,
        min_periods=min_periods,
        null_policy="omit",
        alignment="trailing",
        reducer=lambda x: float(np.average(x, weights=np.arange(1, len(x) + 1))),
    )


def sma_cn(
    values: npt.ArrayLike,
    window: int,
    weight: int = 1,
    *,
    min_periods: int = 1,
) -> Array:
    """Chinese indicator SMA(X,N,M): Y[t]=(M*X[t]+(N-M)*Y[t-1])/N."""
    if window <= 0 or weight <= 0 or weight > window or min_periods <= 0:
        raise ValueError("SMA window/weight/min_periods must satisfy 0 < weight <= window")
    matrix, squeezed = _matrix(values)
    result = np.full(matrix.shape, np.nan)
    alpha = weight / window
    for column in range(matrix.shape[1]):
        state = np.nan
        count = 0
        for row, value in enumerate(matrix[:, column]):
            if not np.isfinite(value):
                continue
            count += 1
            state = value if not np.isfinite(state) else alpha * value + (1 - alpha) * state
            if count >= min_periods:
                result[row, column] = state
    return _restore(result, squeezed)


def decay_linear(values: npt.ArrayLike, window: int, *, min_periods: int | None = None) -> Array:
    return _rolling(
        values,
        window=window,
        min_periods=min_periods,
        null_policy="omit",
        alignment="trailing",
        reducer=lambda x: float(np.average(x, weights=np.arange(1, len(x) + 1))),
    )


def argmax(values: npt.ArrayLike, window: int, *, min_periods: int | None = None) -> Array:
    return _rolling(
        values,
        window=window,
        min_periods=min_periods,
        null_policy="omit",
        alignment="trailing",
        reducer=lambda x: float(np.argmax(x) + 1),
    )


def argmin(values: npt.ArrayLike, window: int, *, min_periods: int | None = None) -> Array:
    return _rolling(
        values,
        window=window,
        min_periods=min_periods,
        null_policy="omit",
        alignment="trailing",
        reducer=lambda x: float(np.argmin(x) + 1),
    )


def regression_slope(
    values: npt.ArrayLike, window: int, *, min_periods: int | None = None
) -> Array:
    minimum = _validate(window, min_periods, "omit", "trailing")
    if minimum == window:
        matrix, squeezed = _matrix(values)
        finite = np.isfinite(matrix)
        filled = np.where(finite, matrix, 0)
        row_index = np.arange(len(matrix), dtype=float)[:, None]

        def trailing_sum(samples: Array) -> Array:
            cumulative = np.vstack((np.zeros((1, matrix.shape[1])), np.cumsum(samples, axis=0)))
            ends = np.arange(1, len(matrix) + 1)
            starts = np.maximum(ends - window, 0)
            return np.asarray(cumulative[ends] - cumulative[starts], dtype=np.float64)

        counts = trailing_sum(finite.astype(float))
        sum_y = trailing_sum(filled)
        absolute_xy = trailing_sum(filled * row_index)
        starts = np.maximum(np.arange(1, len(matrix) + 1) - window, 0)[:, None]
        sum_xy = absolute_xy - starts * sum_y
        sum_x = window * (window - 1) / 2
        sum_x2 = window * (window - 1) * (2 * window - 1) / 6
        denominator = window * sum_x2 - sum_x * sum_x
        result = np.where(
            counts == window,
            (window * sum_xy - sum_x * sum_y) / denominator,
            np.nan,
        )
        return _restore(result, squeezed)
    return _rolling(
        values,
        window=window,
        min_periods=min_periods,
        null_policy="omit",
        alignment="trailing",
        reducer=lambda y: float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0]),
    )


def regression_residual(
    values: npt.ArrayLike, window: int, *, min_periods: int | None = None
) -> Array:
    def residual(y: Array) -> float:
        x = np.arange(len(y), dtype=float)
        coefficients = np.polyfit(x, y, 1)
        return float(y[-1] - np.polyval(coefficients, x[-1]))

    return _rolling(
        values,
        window=window,
        min_periods=min_periods,
        null_policy="omit",
        alignment="trailing",
        reducer=residual,
    )


def count_if(condition: npt.ArrayLike, window: int, *, min_periods: int | None = None) -> Array:
    return rolling_sum(
        np.asarray(condition, dtype=np.float64),
        window,
        min_periods=min_periods,
        null_policy="omit",
    )


def days_since_high(values: npt.ArrayLike, window: int, *, min_periods: int | None = None) -> Array:
    return _rolling(
        values,
        window=window,
        min_periods=min_periods,
        null_policy="omit",
        alignment="trailing",
        reducer=lambda x: float(len(x) - 1 - np.argmax(x)),
    )


def days_since_low(values: npt.ArrayLike, window: int, *, min_periods: int | None = None) -> Array:
    return _rolling(
        values,
        window=window,
        min_periods=min_periods,
        null_policy="omit",
        alignment="trailing",
        reducer=lambda x: float(len(x) - 1 - np.argmin(x)),
    )
