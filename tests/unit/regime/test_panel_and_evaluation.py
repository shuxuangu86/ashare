from datetime import date, timedelta

import numpy as np
import pytest

from aquant.regime import core_state_registry
from aquant.regime.definitions import MarketStateStatus
from aquant.regime.episodes import detect_episodes
from aquant.regime.evaluation import correlation_matrix, evaluate_state, forward_returns
from aquant.regime.panel import MarketStatePanelInput, compute_market_state_panel
from aquant.regime.reporting import correlation_clusters
from aquant.regime.universe import build_dynamic_universes, build_snapshot_membership


def _panel() -> MarketStatePanelInput:
    rows, columns = 300, 12
    dates = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(rows))
    generator = np.random.default_rng(7)
    close = 10 * np.cumprod(1 + generator.normal(0.0003, 0.01, (rows, columns)), axis=0)
    market_cap = np.tile(np.linspace(1, 12, columns), (rows, 1)) * 1e8
    pb = np.tile(np.linspace(1, 4, columns), (rows, 1))
    turnover = np.tile(np.linspace(0.01, 0.05, columns), (rows, 1))
    dividend = np.tile(np.linspace(0.01, 0.04, columns), (rows, 1))
    profit = np.tile(np.linspace(-10, 30, columns), (rows, 1))
    debt = np.tile(np.linspace(20, 80, columns), (rows, 1))
    amount = market_cap * turnover
    universes = dict(
        build_dynamic_universes(
            dates,
            close=close,
            total_market_cap=market_cap,
            pb=pb,
            turnover_rate=turnover,
            dividend_yield=dividend,
            netprofit_yoy=profit,
            microcap_count=4,
        )
    )
    for name in ("HS300", "CSI500", "CSI1000"):
        universes[name] = universes["ALL_A"]
    levels = {name: np.mean(close, axis=1) for name in ("HS300", "CSI500", "CSI1000", "CHINEXT")}
    return MarketStatePanelInput(
        dates,
        tuple(f"{index:06d}.SZ" for index in range(columns)),
        close,
        amount,
        market_cap,
        market_cap,
        pb,
        turnover,
        dividend,
        profit,
        debt,
        close * 1.1,
        close * 0.9,
        universes,
        levels,
    )


def test_panel_engine_covers_every_state_claimed_computable() -> None:
    states = compute_market_state_panel(_panel())
    registry = core_state_registry()
    expected = {
        spec.state_id
        for spec in registry
        if spec.status in {MarketStateStatus.IMPLEMENTED, MarketStateStatus.PARTIAL}
    }
    assert expected <= states.keys()
    assert np.isfinite(states["growth_minus_value_return_40d"][-1])
    assert np.isfinite(states["all_a_above_ma20_ratio"][-1])
    assert np.isfinite(states["microcap_pb_median"][-1])
    assert np.isfinite(states["microcap_crowding_score_v1"][-1])


def test_market_states_are_future_invariant_under_prefix_recomputation() -> None:
    panel = _panel()
    cutoff = 240
    prefix = MarketStatePanelInput(
        dates=panel.dates[:cutoff],
        ts_codes=panel.ts_codes,
        close=panel.close[:cutoff],
        amount=panel.amount[:cutoff],
        total_market_cap=panel.total_market_cap[:cutoff],
        float_market_cap=panel.float_market_cap[:cutoff],
        pb=panel.pb[:cutoff],
        turnover_rate=panel.turnover_rate[:cutoff],
        dividend_yield=panel.dividend_yield[:cutoff],
        netprofit_yoy=panel.netprofit_yoy[:cutoff],
        debt_to_assets=panel.debt_to_assets[:cutoff],
        up_limit=panel.up_limit[:cutoff],
        down_limit=panel.down_limit[:cutoff],
        universes={name: values[:cutoff] for name, values in panel.universes.items()},
        index_levels={name: values[:cutoff] for name, values in panel.index_levels.items()},
    )
    full_states = compute_market_state_panel(panel)
    prefix_states = compute_market_state_panel(prefix)
    for state_id in (
        "growth_minus_value_return_40d",
        "all_a_above_ma20_ratio",
        "microcap_pb_percentile_3y",
        "microcap_crowding_score_v1",
    ):
        np.testing.assert_allclose(
            full_states[state_id][:cutoff],
            prefix_states[state_id],
            equal_nan=True,
        )


def test_dynamic_universes_are_lagged_and_monthly_fixed() -> None:
    panel = _panel()
    microcap = panel.universes["MICROCAP_FIXED_MONTHLY"]
    assert not np.any(microcap[0])
    for index in range(2, len(panel.dates)):
        if panel.dates[index].month == panel.dates[index - 1].month:
            np.testing.assert_array_equal(microcap[index], microcap[index - 1])


def test_historical_snapshot_membership_ignores_incomplete_future_snapshots() -> None:
    dates = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(5))
    result = build_snapshot_membership(
        dates,
        ("A", "B", "C"),
        {
            date(2025, 12, 31): ("A", "B"),
            date(2026, 1, 2): ("C",),
            date(2026, 1, 3): ("B", "C"),
        },
        minimum_constituents=2,
    )
    assert result[0].tolist() == [True, True, False]
    assert result[2].tolist() == [True, True, False]
    assert result[3].tolist() == [False, True, True]


def test_episode_and_evaluation_are_contiguous_and_forward_only() -> None:
    dates = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(8))
    values = np.asarray([0.2, 0.04, 0.03, 0.02, 0.2, 0.01, 0.2, 0.2])
    episodes = detect_episodes(dates, values, state_id="cheap", lower_threshold=0.05)
    assert [episode.duration for episode in episodes] == [3, 1]
    assert episodes[0].side == "LOWER"
    assert episodes[0].recovery_confirmation_date == dates[4]
    level = np.asarray([100.0, 101.0, 102.0, 104.0, 103.0, 106.0, 107.0, 108.0])
    future = forward_returns(level, 2)
    assert future[0] == pytest.approx(0.02)
    short = evaluate_state("cheap", values, future, horizon=2)
    assert short.observations == 6
    assert short.pearson_correlation is None


def test_evaluation_reports_conditional_buckets_and_hac_significance() -> None:
    state = np.linspace(-1.0, 1.0, 200)
    future = state * 0.02 + np.sin(np.arange(200)) * 0.001
    result = evaluate_state("trend", state, future, horizon=20, target_id="HS300_INDEX")
    assert result.target_id == "HS300_INDEX"
    assert result.observations == 200
    assert result.conditional_spread is not None
    assert result.conditional_spread > 0
    assert result.newey_west_t is not None
    assert result.direction_hit_rate > 0.95


def test_correlation_clusters_retain_highly_correlated_states() -> None:
    base = np.arange(100, dtype=np.float64)
    rows = correlation_matrix({"a": base, "b": base * 2, "c": -base}, minimum_observations=20)
    clusters = correlation_clusters(rows, threshold=0.95)
    assert any(set(component) == {"a", "b", "c"} for component in clusters.values())
