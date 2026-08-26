from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from aquant.backtest import AshareOpenMatcher, MarketSession, NextOpenMatcher
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar, LimitStatus, SecurityStatus
from aquant.strategies.microcap import (
    CostScenario,
    MicrocapExperimentRunner,
    MicrocapExperimentSpec,
    MicrocapObservation,
    MicrocapPrototype,
    MicrocapSnapshot,
    RebalanceFrequency,
    build_experiment_matrix,
    generate_rebalance_dates,
    matcher_for_cost_scenario,
    render_microcap_matrix_report,
)


def test_matrix_has_72_unique_runs_and_separates_capital_profiles() -> None:
    matrix = build_experiment_matrix()

    assert len(matrix) == 72
    assert len({spec.run_id for spec in matrix}) == 72
    capital_by_prototype = {
        prototype: {spec.initial_equity for spec in matrix if spec.prototype is prototype}
        for prototype in MicrocapPrototype
    }
    assert capital_by_prototype[MicrocapPrototype.SMALLEST_400] == {Decimal("100000000")}
    assert capital_by_prototype[MicrocapPrototype.EXECUTABLE_95] == {Decimal("1000000")}


def test_rebalance_dates_use_complete_trading_periods_and_leave_a_fill_session() -> None:
    dates = (
        date(2026, 7, 27),
        date(2026, 7, 28),
        date(2026, 7, 29),
        date(2026, 7, 30),
        date(2026, 7, 31),
        date(2026, 8, 3),
        date(2026, 8, 4),
    )

    assert generate_rebalance_dates(dates, RebalanceFrequency.DAILY) == frozenset(dates[:-1])
    assert generate_rebalance_dates(dates, RebalanceFrequency.WEEKLY) == frozenset(
        {date(2026, 7, 31)}
    )
    assert generate_rebalance_dates(dates, RebalanceFrequency.MONTHLY) == frozenset(
        {date(2026, 7, 31)}
    )
    with pytest.raises(ValueError, match="at least two"):
        generate_rebalance_dates((dates[0],), RebalanceFrequency.DAILY)
    with pytest.raises(ValueError, match="strictly increasing"):
        generate_rebalance_dates((dates[1], dates[0]), RebalanceFrequency.DAILY)


def test_cost_scenarios_select_frictionless_or_rule_aware_matchers() -> None:
    assert isinstance(matcher_for_cost_scenario(CostScenario.FRICTIONLESS), NextOpenMatcher)
    for scenario in (
        CostScenario.RULES_ONLY,
        CostScenario.BASE_COST,
        CostScenario.DOUBLE_COST,
    ):
        assert isinstance(matcher_for_cost_scenario(scenario), AshareOpenMatcher)


def _market_fixture() -> tuple[tuple[MarketSession, ...], dict[date, MicrocapSnapshot]]:
    dates = (date(2026, 7, 16), date(2026, 7, 17))
    symbols = tuple(Symbol.parse(f"{600000 + rank:06d}.XSHG") for rank in range(1, 402))
    sessions: list[MarketSession] = []
    snapshots: dict[date, MicrocapSnapshot] = {}
    for trade_date in dates:
        asof = datetime(trade_date.year, trade_date.month, trade_date.day, 7, tzinfo=UTC)
        bars = tuple(
            DailyBar(
                symbol,
                trade_date,
                Decimal("10"),
                Decimal("10"),
                Decimal("10"),
                Decimal("10"),
                Decimal("10000000"),
                Decimal("100000000"),
            )
            for symbol in symbols
        )
        statuses = tuple(
            SecurityStatus(
                symbol,
                trade_date,
                False,
                False,
                LimitStatus.NONE,
                Decimal("10000000"),
            )
            for symbol in symbols
        )
        sessions.append(
            MarketSession(
                trade_date,
                datetime(
                    trade_date.year,
                    trade_date.month,
                    trade_date.day,
                    1,
                    30,
                    tzinfo=UTC,
                ),
                asof,
                bars,
                statuses,
            )
        )
        snapshots[trade_date] = MicrocapSnapshot(
            trade_date,
            asof,
            tuple(
                MicrocapObservation(
                    symbol,
                    trade_date,
                    asof,
                    date(2020, 1, 1),
                    Decimal(rank),
                    False,
                    False,
                )
                for rank, symbol in enumerate(symbols, start=1)
            ),
        )
    return tuple(sessions), snapshots


def test_runner_executes_t_to_next_open_and_report_preserves_protocol() -> None:
    sessions, snapshots = _market_fixture()
    spec = MicrocapExperimentSpec(
        MicrocapPrototype.SMALLEST_100,
        RebalanceFrequency.DAILY,
        CostScenario.FRICTIONLESS,
        Decimal("100000000"),
    )

    outcome = MicrocapExperimentRunner(sessions=sessions, snapshots=snapshots).run_one(spec)
    report = render_microcap_matrix_report((outcome,))

    assert len(outcome.result.fills) == 100
    assert all(fill.occurred_at == sessions[1].open_at for fill in outcome.result.fills)
    assert "smallest-100" in report
    assert "100000000" in report
    assert "RMB 1,000,000 executable-capital profile" in report


def test_runner_and_report_reject_empty_inputs() -> None:
    with pytest.raises(ValueError, match="requires sessions"):
        MicrocapExperimentRunner(sessions=(), snapshots={})
    with pytest.raises(ValueError, match="requires outcomes"):
        render_microcap_matrix_report(())
    sessions, snapshots = _market_fixture()
    runner = MicrocapExperimentRunner(sessions=sessions, snapshots=snapshots)
    with pytest.raises(ValueError, match="cannot be empty"):
        runner.run_all(())
    with pytest.raises(ValueError, match="initial equity"):
        MicrocapExperimentSpec(
            MicrocapPrototype.SMALLEST_100,
            RebalanceFrequency.DAILY,
            CostScenario.BASE_COST,
            Decimal("0"),
        )
