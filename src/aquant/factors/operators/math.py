from collections.abc import Callable

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
ArrayLike = npt.ArrayLike


def _unary(values: ArrayLike, operation: Callable[[Array], Array]) -> Array:
    resolved = np.asarray(values, dtype=np.float64)
    with np.errstate(all="ignore"):
        result = operation(resolved)
    return np.where(np.isfinite(result), result, np.nan)


def safe_div(numerator: ArrayLike, denominator: ArrayLike) -> Array:
    left, right = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    result = np.full(left.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right) & (right != 0)
    np.divide(left, right, out=result, where=valid)
    result[~np.isfinite(result)] = np.nan
    return result


def signed_power(values: ArrayLike, exponent: float) -> Array:
    if not np.isfinite(exponent):
        raise ValueError("signed power exponent must be finite")
    resolved = np.asarray(values, dtype=np.float64)
    with np.errstate(all="ignore"):
        result = np.sign(resolved) * np.power(np.abs(resolved), exponent)
    return np.where(np.isfinite(result), result, np.nan)


def abs(values: ArrayLike) -> Array:
    return _unary(values, np.abs)


def sign(values: ArrayLike) -> Array:
    return _unary(values, np.sign)


def log(values: ArrayLike) -> Array:
    resolved = np.asarray(values, dtype=np.float64)
    return _unary(resolved, lambda item: np.where(item > 0, np.log(item), np.nan))


def log1p_abs(values: ArrayLike) -> Array:
    return _unary(values, lambda item: np.log1p(np.abs(item)))


def sqrt(values: ArrayLike) -> Array:
    resolved = np.asarray(values, dtype=np.float64)
    return _unary(resolved, lambda item: np.where(item >= 0, np.sqrt(item), np.nan))


def clip(values: ArrayLike, lower: float, upper: float) -> Array:
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("clip bounds must be finite and ordered")
    return _unary(values, lambda item: np.clip(item, lower, upper))


def minimum(left: ArrayLike, right: ArrayLike) -> Array:
    with np.errstate(all="ignore"):
        result = np.minimum(
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
        )
    return np.where(np.isfinite(result), result, np.nan)


def maximum(left: ArrayLike, right: ArrayLike) -> Array:
    with np.errstate(all="ignore"):
        result = np.maximum(
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
        )
    return np.where(np.isfinite(result), result, np.nan)


def where(condition: npt.ArrayLike, if_true: ArrayLike, if_false: ArrayLike) -> Array:
    result = np.where(
        np.asarray(condition, dtype=bool),
        np.asarray(if_true, dtype=np.float64),
        np.asarray(if_false, dtype=np.float64),
    )
    return np.where(np.isfinite(result), result, np.nan)
