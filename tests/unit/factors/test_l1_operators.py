import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from aquant.factors.operators import math
from aquant.factors.operators.cross_sectional import (
    cs_percentile,
    cs_rank,
    cs_zscore,
    group_rank,
    winsorize_mad,
    winsorize_quantile,
)
from aquant.factors.operators.neutralization import (
    industry_neutralize,
    multi_exposure_neutralize,
)
from aquant.factors.operators.regression import linear_regression
from aquant.factors.operators.time_series import (
    argmax,
    count_if,
    days_since_high,
    decay_linear,
    delay,
    delta,
    ema,
    regression_residual,
    regression_slope,
    returns,
    rolling_corr,
    rolling_cov,
    rolling_kurt,
    rolling_max,
    rolling_mean,
    rolling_median,
    rolling_min,
    rolling_quantile,
    rolling_rank,
    rolling_skew,
    rolling_std,
    rolling_sum,
    rolling_var,
    ts_zscore,
)


def test_math_operators_handle_zero_nonfinite_and_domains() -> None:
    values = np.array([-4.0, 0.0, 4.0, np.inf, np.nan])
    np.testing.assert_allclose(
        math.safe_div(values, np.array([2, 0, 2, 1, 1])),
        [-2, np.nan, 2, np.nan, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(math.signed_power(values, 0.5), [-2, 0, 2, np.nan, np.nan])
    np.testing.assert_allclose(math.sqrt(values), [np.nan, 0, 2, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(
        math.log(np.array([-1, 1, np.e])), [np.nan, 0, 1], atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        math.where([True, False], [1, 2], [3, np.inf]), [1.0, np.nan], equal_nan=True
    )
    with pytest.raises(ValueError):
        math.clip(values, 2, 1)


def test_time_series_small_sample_and_boundaries() -> None:
    values = np.array([1.0, 2.0, 4.0, 8.0])
    np.testing.assert_allclose(delay(values), [np.nan, 1, 2, 4], equal_nan=True)
    np.testing.assert_allclose(delta(values), [np.nan, 1, 2, 4], equal_nan=True)
    np.testing.assert_allclose(returns(values), [np.nan, 1, 1, 1], equal_nan=True)
    np.testing.assert_allclose(
        rolling_sum(values, 3, min_periods=2), [np.nan, 3, 7, 14], equal_nan=True
    )
    np.testing.assert_allclose(
        rolling_mean(values, 2), pd.Series(values).rolling(2).mean().to_numpy()
    )
    np.testing.assert_allclose(
        rolling_std(values, 3), pd.Series(values).rolling(3).std().to_numpy(), equal_nan=True
    )
    np.testing.assert_allclose(
        rolling_var(values, 3), pd.Series(values).rolling(3).var().to_numpy(), equal_nan=True
    )
    np.testing.assert_allclose(rolling_min(values, 2), [np.nan, 1, 2, 4], equal_nan=True)
    np.testing.assert_allclose(rolling_max(values, 2), [np.nan, 2, 4, 8], equal_nan=True)
    np.testing.assert_allclose(rolling_median(values, 2), [np.nan, 1.5, 3, 6], equal_nan=True)
    np.testing.assert_allclose(
        rolling_quantile(values, 2, 0.25), [np.nan, 1.25, 2.5, 5], equal_nan=True
    )
    np.testing.assert_allclose(rolling_rank(values, 3), [np.nan, np.nan, 5 / 6, 5 / 6])
    with pytest.raises(ValueError):
        rolling_mean(values, 2, alignment="centered")  # type: ignore[arg-type]


def test_pair_moments_shape_null_policy_and_statistics() -> None:
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    assert rolling_corr(x, y, 3, min_periods=2)[-1] == pytest.approx(1)
    assert rolling_cov(x, y, 3, min_periods=2)[-1] == pytest.approx(1)
    assert np.isnan(rolling_corr(x, y, 3, min_periods=2, null_policy="propagate")[3])
    symmetric = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert rolling_skew(symmetric, 5)[-1] == pytest.approx(0)
    assert rolling_kurt(symmetric, 5)[-1] == pytest.approx(-1.3)


def test_derived_time_series_operators_are_hand_verifiable() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(ts_zscore(values, 2), [np.nan, 1, 1, 1], equal_nan=True)
    np.testing.assert_allclose(ema(values, 3), [1, 1.5, 2.25, 3.125])
    np.testing.assert_allclose(decay_linear(values, 3), [np.nan, np.nan, 14 / 6, 20 / 6])
    np.testing.assert_allclose(argmax(values, 3), [np.nan, np.nan, 3, 3])
    np.testing.assert_allclose(days_since_high(values, 3), [np.nan, np.nan, 0, 0])
    np.testing.assert_allclose(regression_slope(values, 3), [np.nan, np.nan, 1, 1])
    np.testing.assert_allclose(regression_residual(values, 3), [np.nan, np.nan, 0, 0], atol=1e-12)
    np.testing.assert_allclose(count_if(values > 2, 2), [np.nan, 0, 1, 2], equal_nan=True)


def test_future_mutation_never_changes_past_rolling_output() -> None:
    original = np.arange(1.0, 31.0)
    changed = original.copy()
    changed[20:] = 1_000_000
    functions = (
        lambda x: rolling_mean(x, 5),
        lambda x: rolling_std(x, 5),
        lambda x: regression_slope(x, 5),
        lambda x: decay_linear(x, 5),
        lambda x: ts_zscore(x, 5),
    )
    for function in functions:
        np.testing.assert_allclose(function(original)[:20], function(changed)[:20], equal_nan=True)


def test_cross_sectional_rank_winsorization_and_groups() -> None:
    values = np.array([3.0, 1.0, 1.0, np.nan])
    np.testing.assert_allclose(cs_rank(values), [3, 1.5, 1.5, np.nan], equal_nan=True)
    np.testing.assert_allclose(cs_percentile(values), [5 / 6, 1 / 3, 1 / 3, np.nan])
    zscores = cs_zscore(np.array([1.0, 2.0, 3.0]))
    assert np.mean(zscores) == pytest.approx(0)
    assert np.std(zscores) == pytest.approx(1)
    grouped = group_rank(np.array([2.0, 1.0, 4.0, 3.0]), np.array(["A", "A", "B", "B"]))
    np.testing.assert_allclose(grouped, [0.75, 0.25, 0.75, 0.25])
    assert np.max(winsorize_quantile(np.array([0.0, 1.0, 100.0]), lower=0, upper=0.5)) == 1
    assert np.max(winsorize_mad(np.array([0.0, 1.0, 100.0]), scale=1)) < 100


def test_weighted_neutralization_and_regression_remove_exposure() -> None:
    size = np.arange(1.0, 8.0)
    industry = np.array(["A", "A", "A", "B", "B", "B", "B"])
    values = 3 + 2 * size + (industry == "B") * 5
    residuals = industry_neutralize(values, industry)
    assert abs(np.mean(residuals)) < 1e-10
    residuals = multi_exposure_neutralize(values, size[:, None])
    assert abs(np.corrcoef(residuals, size)[0, 1]) < 1e-10
    result = linear_regression(values, size[:, None])
    slope, intercept = np.polyfit(size, values, 1)
    np.testing.assert_allclose(result.coefficients, [intercept, slope], atol=1e-10)
    assert np.count_nonzero(result.valid) == len(size)


@given(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=50,
        unique=True,
    )
)
def test_rank_is_monotonic(values: list[float]) -> None:
    resolved = np.asarray(values)
    ranks = cs_rank(resolved)
    order = np.argsort(resolved)
    assert np.all(np.diff(ranks[order]) > 0)
