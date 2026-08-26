from datetime import date, timedelta

import numpy as np
import pytest

from aquant.factors.evaluation import (
    basic_style_exposures,
    evaluate_institutional,
    historical_market_regimes,
    newey_west_mean_t,
    pit_forward_return_labels,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, tuple[date, ...]]:
    rows, columns = 90, 20
    time = np.arange(rows, dtype=float)[:, None]
    security = np.arange(columns, dtype=float)[None, :]
    factor = security + np.sin(time / 5 + security / 3)
    close = 10 + time * (0.02 + security / 1000) + security / 2
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(rows))
    return factor, close, dates


def test_pit_labels_start_after_factor_date_and_ignore_factor_day_price() -> None:
    _, close, dates = _fixture()
    labels, timings = pit_forward_return_labels(close, dates, 5)
    assert timings[0].factor_date == dates[0]
    assert timings[0].execution_date == timings[0].return_start == dates[1]
    assert timings[0].return_end == dates[6]
    changed = close.copy()
    changed[0] *= 100
    changed_labels, _ = pit_forward_return_labels(changed, dates, 5)
    np.testing.assert_allclose(labels, changed_labels, equal_nan=True)
    with pytest.raises(ValueError, match="overlap"):
        timings[0].__class__(
            dates[0],
            timings[0].available_at,
            dates[0],
            dates[0],
            dates[5],
        )


def test_institutional_metrics_include_time_slices_costs_and_exposures() -> None:
    factor, close, dates = _fixture()
    labels, _ = pit_forward_return_labels(close, dates, 5)
    market_cap = np.broadcast_to(np.arange(1, 21, dtype=float) * 1e9, close.shape)
    turnover = 1 + np.abs(np.sin(np.arange(90)[:, None] / 5 + np.arange(20)[None, :]))
    exposures = basic_style_exposures(close, market_cap, turnover, window=20)
    evaluation = evaluate_institutional(
        factor,
        labels,
        dates,
        horizon=5,
        cost_bps=10,
        regimes=historical_market_regimes(close),
        exposures=exposures,
    )
    assert evaluation.observations > 1000
    assert np.isfinite(evaluation.rank_ic_mean)
    assert np.isfinite(evaluation.newey_west_t)
    assert evaluation.annual_rank_ic
    assert evaluation.quarterly_rank_ic
    assert evaluation.regime_rank_ic
    assert {name for name, _ in evaluation.style_exposures} == {
        "beta",
        "liquidity",
        "price",
        "size",
        "volatility",
    }
    assert evaluation.net_long_short_return <= evaluation.long_short_return


def test_newey_west_and_evaluation_validation_fail_closed() -> None:
    assert np.isfinite(newey_west_mean_t(np.arange(20, dtype=float)))
    with pytest.raises(ValueError, match="lag"):
        newey_west_mean_t(np.arange(20, dtype=float), max_lag=-1)
    factor, close, dates = _fixture()
    labels, _ = pit_forward_return_labels(close, dates, 5)
    with pytest.raises(ValueError, match="regimes"):
        evaluate_institutional(factor, labels, dates, horizon=5, regimes=("BULL",))
