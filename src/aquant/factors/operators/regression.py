from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class RegressionResult:
    coefficients: npt.NDArray[np.float64]
    residuals: npt.NDArray[np.float64]
    valid: npt.NDArray[np.bool_]


def linear_regression(
    target: npt.ArrayLike,
    design: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
    add_intercept: bool = True,
) -> RegressionResult:
    from aquant.factors.operators.neutralization import weighted_residualize

    y = np.asarray(target, dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    residuals = weighted_residualize(y, x, weights=weights, add_intercept=add_intercept)
    valid = np.isfinite(residuals)
    fitted_design = x[valid]
    if add_intercept:
        fitted_design = np.column_stack((np.ones(np.count_nonzero(valid)), fitted_design))
    resolved_weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    root = np.sqrt(resolved_weights[valid])
    coefficients = np.linalg.lstsq(fitted_design * root[:, None], y[valid] * root, rcond=None)[0]
    return RegressionResult(coefficients, residuals, valid)
