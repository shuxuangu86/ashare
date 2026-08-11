from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from scripts.run_nested_l4_backtest import (
    _BASE_COST_ASSUMPTIONS,
    _daily_curve_csv,
    _fold_performance,
    _snapshots,
)

from aquant.domain.identifiers import Symbol
from aquant.strategies.microcap.experiments import RebalanceFrequency
from aquant.strategies.nested_l4 import select_l4_configuration


def test_l4_configuration_uses_only_supplied_inner_validation() -> None:
    generator = np.random.default_rng(12)
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(80))
    scores = generator.normal(size=(80, 40))
    close = np.full((80, 40), 100.0)
    returns = 0.01 * np.sign(scores) + generator.normal(scale=0.002, size=scores.shape)
    for index in range(1, len(close)):
        close[index] = close[index - 1] * (1 + returns[index - 1])
    validation = np.arange(20, 60, dtype=np.int64)
    first = select_l4_configuration(
        scores=scores,
        close=close,
        trade_dates=dates,
        validation_positions=validation,
        target_counts=(10, 20),
        frequencies=(RebalanceFrequency.WEEKLY, RebalanceFrequency.MONTHLY),
    )
    changed = close.copy()
    changed[60:] *= 20
    second = select_l4_configuration(
        scores=scores,
        close=changed,
        trade_dates=dates,
        validation_positions=validation,
        target_counts=(10, 20),
        frequencies=(RebalanceFrequency.WEEKLY, RebalanceFrequency.MONTHLY),
    )

    assert first == second
    assert first["outer_test_used_for_selection"] is False
    assert len(first["candidates"]) == 4
    assert first["trial_count"] == 4
    assert first["execution_model"].endswith("CONSERVATIVE_PROXY")
    assert first["benchmark_model"].endswith("CLOSE_RETURN_PROXY")

    default_search = select_l4_configuration(
        scores=scores,
        close=close,
        trade_dates=dates,
        validation_positions=validation,
    )
    assert default_search["trial_count"] == 10
    assert {candidate["target_count"] for candidate in default_search["candidates"]} == {
        20,
        30,
        50,
        80,
        100,
    }
    assert all("target_met" in candidate for candidate in default_search["candidates"])


def test_outer_fold_performance_is_reported_separately() -> None:
    dates = tuple(date(2021, 1, day) for day in range(1, 5))
    result = _fold_performance(
        folds=[
            {"fold": 1, "test_start": "2021-01-01", "test_end": "2021-01-02"},
            {"fold": 2, "test_start": "2021-01-03", "test_end": "2021-01-04"},
        ],
        aligned_dates=dates,
        strategy_returns=dict(zip(dates, (0.01, 0.01, 0.00, 0.00), strict=True)),
        benchmark_returns=dict(zip(dates, (0.00, 0.00, 0.01, 0.02), strict=True)),
    )

    assert [fold["sessions"] for fold in result] == [2, 2]
    assert result[0]["annual_excess_return"] > 0
    assert result[1]["annual_excess_return"] < 0
    assert result[0]["relative_maximum_drawdown"] <= 0
    assert result[1]["tracking_error"] > 0

    curve = _daily_curve_csv(
        folds=[
            {"fold": 1, "test_start": "2021-01-01", "test_end": "2021-01-02"},
            {"fold": 2, "test_start": "2021-01-03", "test_end": "2021-01-04"},
        ],
        aligned_dates=dates,
        strategy_returns=dict(zip(dates, (0.01, 0.01, 0.00, 0.00), strict=True)),
        benchmark_returns=dict(zip(dates, (0.00, 0.00, 0.01, 0.02), strict=True)),
    )
    rows = curve.splitlines()
    assert len(rows) == 5
    assert rows[1].startswith("2021-01-01,1,")
    assert rows[-1].startswith("2021-01-04,2,")


def test_nested_backtest_discloses_executable_cost_assumptions() -> None:
    assert _BASE_COST_ASSUMPTIONS["execution_time"] == "T_PLUS_1_OPEN"
    assert _BASE_COST_ASSUMPTIONS["slippage_bps_each_fill"] == 5
    assert _BASE_COST_ASSUMPTIONS["minimum_commission_cny"] == 5
    assert _BASE_COST_ASSUMPTIONS[
        "maximum_prior_20d_average_volume_participation"
    ] == pytest.approx(0.005)


def test_l4_snapshots_only_require_full_universe_on_rebalance_dates() -> None:
    first = date(2021, 1, 4)
    second = date(2021, 1, 5)
    close_at = datetime(2021, 1, 4, 15, tzinfo=UTC)
    symbol = Symbol.parse("600001.XSHG")
    observation = SimpleNamespace(
        symbol=symbol,
        list_date=date(2020, 1, 1),
        suspended=False,
        is_st=False,
        is_delisting_risk=False,
    )
    snapshots = _snapshots(
        trading_dates=(first, second),
        risk_snapshots={first: SimpleNamespace(asof_time=close_at, observations=(observation,))},
        scores=np.zeros((2, 1)),
        date_index={first: 0, second: 1},
        code_index={"600001.SH": 0},
    )

    assert tuple(snapshots) == (first,)
    assert len(snapshots[first].observations) == 1
