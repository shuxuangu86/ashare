from datetime import UTC, date, datetime
from decimal import Decimal

from aquant.backtest import (
    AshareExecutionRules,
    AshareOpenMatcher,
    EventDrivenBacktest,
    MarketSession,
    UnfilledOrderPolicy,
)
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar, LimitStatus, SecurityStatus
from aquant.strategies.microcap import (
    MicrocapEqualWeightStrategy,
    MicrocapObservation,
    MicrocapSnapshot,
    MicrocapUniverseConfig,
)

SYMBOLS = (Symbol.parse("600001.XSHG"), Symbol.parse("000002.XSHE"))


def _session(day: int) -> MarketSession:
    trade_date = date(2026, 7, day)
    bars = tuple(
        DailyBar(
            symbol,
            trade_date,
            Decimal(price),
            Decimal(price),
            Decimal(price),
            Decimal(price),
            Decimal("1000000"),
            Decimal("10000000"),
        )
        for symbol, price in zip(SYMBOLS, ("10", "20"), strict=True)
    )
    statuses = tuple(
        SecurityStatus(symbol, trade_date, False, False, LimitStatus.NONE) for symbol in SYMBOLS
    )
    return MarketSession(
        trade_date,
        datetime(2026, 7, day, 1, 30, tzinfo=UTC),
        datetime(2026, 7, day, 11, 30, tzinfo=UTC),
        bars,
        statuses,
    )


def test_microcap_rank_at_t_fills_only_at_next_open() -> None:
    signal_date = date(2026, 7, 16)
    asof = datetime(2026, 7, 16, 11, 30, tzinfo=UTC)
    observations = tuple(
        MicrocapObservation(
            symbol,
            signal_date,
            asof,
            date(2020, 1, 1),
            Decimal(market_cap),
            False,
            False,
        )
        for symbol, market_cap in zip(SYMBOLS, ("100", "200"), strict=True)
    )
    strategy = MicrocapEqualWeightStrategy(
        snapshots={signal_date: MicrocapSnapshot(signal_date, asof, observations)},
        rebalance_dates={signal_date},
        universe=MicrocapUniverseConfig(target_count=1, minimum_constituents=1),
    )
    result = EventDrivenBacktest(
        run_id="microcap-400-p0",
        initial_cash=Decimal("100000"),
        matcher=AshareOpenMatcher(
            rules=AshareExecutionRules(
                slippage_bps=Decimal("0"),
                max_volume_participation=Decimal("1"),
            )
        ),
        unfilled_order_policy=UnfilledOrderPolicy.CANCEL_AFTER_OPEN,
    ).run((_session(16), _session(17)), strategy)

    assert len(result.orders) == 1
    assert len(result.fills) == 1
    assert result.fills[0].symbol == SYMBOLS[0]
    assert result.fills[0].occurred_at == _session(17).open_at
    assert result.equity_curve[0].positions == ()
    assert result.equity_curve[1].positions[0].symbol == SYMBOLS[0]
