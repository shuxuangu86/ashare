import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def winsorize_mad(values: Array, *, scale: float = 3.0) -> Array:
    if scale <= 0:
        raise ValueError("winsorize scale must be positive")
    resolved = np.asarray(values, dtype=np.float64).copy()
    valid = np.isfinite(resolved)
    sample = resolved[valid]
    if not sample.size:
        return resolved
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    if mad == 0:
        return resolved
    robust_sigma = 1.4826 * mad
    resolved[valid] = np.clip(sample, median - scale * robust_sigma, median + scale * robust_sigma)
    return resolved


def zscore(values: Array) -> Array:
    resolved = np.asarray(values, dtype=np.float64)
    result = np.full(resolved.shape, np.nan)
    valid = np.isfinite(resolved)
    sample = resolved[valid]
    if not sample.size:
        return result
    standard_deviation = float(np.std(sample))
    result[valid] = (
        0.0
        if np.isclose(standard_deviation, 0.0, atol=1e-12)
        else (sample - np.mean(sample)) / standard_deviation
    )
    return result


def neutralize(values: Array, exposures: Array) -> Array:
    target = np.asarray(values, dtype=np.float64)
    design_values = np.asarray(exposures, dtype=np.float64)
    if target.ndim != 1 or design_values.ndim != 2 or design_values.shape[0] != target.size:
        raise ValueError("neutralization requires values (n,) and exposures (n, k)")
    result = np.full(target.shape, np.nan)
    valid = np.isfinite(target) & np.all(np.isfinite(design_values), axis=1)
    if np.count_nonzero(valid) <= design_values.shape[1]:
        return result
    design = np.column_stack((np.ones(np.count_nonzero(valid)), design_values[valid]))
    coefficients = np.linalg.lstsq(design, target[valid], rcond=None)[0]
    result[valid] = target[valid] - design @ coefficients
    return result


def preprocess_cross_section(
    values: Array,
    *,
    exposures: Array | None = None,
    winsorize_scale: float = 3.0,
) -> Array:
    result = winsorize_mad(values, scale=winsorize_scale)
    if exposures is not None:
        result = neutralize(result, exposures)
    return zscore(result)
