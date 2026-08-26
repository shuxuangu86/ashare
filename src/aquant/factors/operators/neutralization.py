import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def weighted_residualize(
    values: npt.ArrayLike,
    exposures: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
    add_intercept: bool = True,
    minimum_observations: int | None = None,
) -> Array:
    target = np.asarray(values, dtype=np.float64)
    design = np.asarray(exposures, dtype=np.float64)
    if target.ndim != 1 or design.ndim != 2 or design.shape[0] != target.size:
        raise ValueError("neutralization requires values (n,) and exposures (n, k)")
    resolved_weights = (
        np.ones(target.size, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if resolved_weights.shape != target.shape:
        raise ValueError("neutralization weights must match values")
    valid = (
        np.isfinite(target)
        & np.all(np.isfinite(design), axis=1)
        & np.isfinite(resolved_weights)
        & (resolved_weights > 0)
    )
    columns = design.shape[1] + int(add_intercept)
    required = max(columns + 1, minimum_observations or 0)
    result = np.full(target.shape, np.nan)
    if np.count_nonzero(valid) < required:
        return result
    x = design[valid]
    if add_intercept:
        x = np.column_stack((np.ones(len(x)), x))
    root_weights = np.sqrt(resolved_weights[valid])
    coefficients = np.linalg.lstsq(
        x * root_weights[:, None], target[valid] * root_weights, rcond=None
    )[0]
    result[valid] = target[valid] - x @ coefficients
    return result


def industry_exposures(groups: npt.ArrayLike) -> Array:
    resolved = np.asarray(groups)
    if resolved.ndim != 1:
        raise ValueError("industry groups must be one-dimensional")
    valid_groups = sorted({str(value) for value in resolved if value is not None and str(value)})
    if len(valid_groups) <= 1:
        return np.empty((len(resolved), 0), dtype=np.float64)
    # Drop the first category to avoid exact collinearity with the intercept.
    return np.column_stack(
        [
            np.asarray([float(str(value) == group) for value in resolved])
            for group in valid_groups[1:]
        ]
    )


def industry_neutralize(
    values: npt.ArrayLike,
    groups: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
    minimum_observations: int | None = None,
) -> Array:
    return weighted_residualize(
        values,
        industry_exposures(groups),
        weights=weights,
        minimum_observations=minimum_observations,
    )


def size_neutralize(
    values: npt.ArrayLike,
    log_float_market_cap: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
) -> Array:
    size = np.asarray(log_float_market_cap, dtype=np.float64)
    return weighted_residualize(values, size[:, None], weights=weights)


def multi_exposure_neutralize(
    values: npt.ArrayLike,
    exposures: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
    minimum_observations: int | None = None,
) -> Array:
    return weighted_residualize(
        values,
        exposures,
        weights=weights,
        minimum_observations=minimum_observations,
    )
