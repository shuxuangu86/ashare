from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from aquant.backtest import MarketSession, StrategyContext
from aquant.backtest.accounting import PortfolioSnapshot, Position
from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar
from aquant.strategies.microcap import (
    MicrocapEqualWeightStrategy,
    MicrocapObservation,
    MicrocapSnapshot,
    MicrocapUniverseConfig,
)
from aquant.strategies.microcap.models import MicrocapSelector

TRADE_DATE = date(2026, 7, 17)
ASOF = datetime(2026, 7, 17, 11, 30, tzinfo=UTC)


def _observation(
    canonical: str,
    market_cap: str,
    *,
    listed_days: int = 500,
    suspended: bool = False,
    is_st: bool = False,
    available_at: datetime = ASOF,
) -> MicrocapObservation:
    return MicrocapObservation(
        Symbol.parse(canonical),
        TRADE_DATE,
        available_at,
        TRADE_DATE.fromordinal(TRADE_DATE.toordinal() - listed_days),
        Decimal(market_cap),
        suspended,
        is_st,
    )


def _snapshot(*observations: MicrocapObservation) -> MicrocapSnapshot:
    return MicrocapSnapshot(TRADE_DATE, ASOF, observations)


def test_selector_is_deterministic_and_reports_every_exclusion() -> None:
    observations = (
        _observation("600003.XSHG", "100"),
        _observation("000002.XSHE", "50"),
        _observation("600001.XSHG", "50"),
        _observation("430001.XBSE", "1"),
        _observation("600004.XSHG", "2", listed_days=30),
        _observation("600005.XSHG", "3", is_st=True),
        _observation("600006.XSHG", "4", suspended=True),
    )
    selection = MicrocapSelector(
        MicrocapUniverseConfig(target_count=2, minimum_constituents=2)
    ).select(_snapshot(*observations))

    assert selection.symbols == (
        Symbol.parse("000002.XSHE"),
        Symbol.parse("600001.XSHG"),
    )
    assert selection.eligible_count == 3
    assert dict(selection.excluded_counts) == {
        "delisting_risk": 0,
        "exchange": 1,
        "new_listing": 1,
        "st": 1,
        "suspended": 1,
    }


def test_snapshot_rejects_lookahead_and_incomplete_universe_fails_closed() -> None:
    future = datetime(2026, 7, 17, 12, tzinfo=UTC)
    with pytest.raises(ValueError, match="unavailable"):
        _snapshot(_observation("600001.XSHG", "10", available_at=future))
    with pytest.raises(ValueError, match="incomplete"):
        MicrocapSelector(MicrocapUniverseConfig(target_count=2, minimum_constituents=2)).select(
            _snapshot(_observation("600001.XSHG", "10"))
        )


def test_equal_weight_strategy_emits_sells_before_buys_and_rounds_lots() -> None:
    selected = Symbol.parse("600001.XSHG")
    removed = Symbol.parse("600002.XSHG")
    snapshot = _snapshot(_observation(str(selected), "10"))
    strategy = MicrocapEqualWeightStrategy(
        snapshots={TRADE_DATE: snapshot},
        rebalance_dates={TRADE_DATE},
        universe=MicrocapUniverseConfig(target_count=1, minimum_constituents=1),
        cash_buffer_weight=Decimal("0.02"),
    )
    bars = tuple(
        DailyBar(
            symbol,
            TRADE_DATE,
            Decimal("10"),
            Decimal("10"),
            Decimal("10"),
            Decimal("10"),
            Decimal("100000"),
            Decimal("1000000"),
        )
        for symbol in (selected, removed)
    )
    session = MarketSession(
        TRADE_DATE,
        datetime(2026, 7, 17, 1, 30, tzinfo=UTC),
        ASOF,
        bars,
    )
    portfolio = PortfolioSnapshot(
        ASOF,
        Decimal("9000"),
        Decimal("1000"),
        Decimal("10000"),
        Decimal("0"),
        (Position(removed, 100, Decimal("10")),),
    )

    orders = strategy.on_close(StrategyContext(session, portfolio))

    assert orders[0].symbol == removed and orders[0].side is Side.SELL
    assert orders[1].symbol == selected and orders[1].side is Side.BUY
    assert orders[1].quantity == 900


def test_strategy_skips_non_rebalance_dates_and_requires_pit_snapshot() -> None:
    symbol = Symbol.parse("600001.XSHG")
    bar = DailyBar(
        symbol,
        TRADE_DATE,
        Decimal("10"),
        Decimal("10"),
        Decimal("10"),
        Decimal("10"),
        Decimal("100000"),
        Decimal("1000000"),
    )
    session = MarketSession(
        TRADE_DATE,
        datetime(2026, 7, 17, 1, 30, tzinfo=UTC),
        ASOF,
        (bar,),
    )
    portfolio = PortfolioSnapshot(
        ASOF, Decimal("10000"), Decimal("0"), Decimal("10000"), Decimal("0"), ()
    )
    context = StrategyContext(session, portfolio)
    assert MicrocapEqualWeightStrategy(snapshots={}, rebalance_dates=set()).on_close(context) == ()
    with pytest.raises(KeyError, match="missing micro-cap PIT"):
        MicrocapEqualWeightStrategy(
            snapshots={},
            rebalance_dates={TRADE_DATE},
            universe=MicrocapUniverseConfig(target_count=1, minimum_constituents=1),
        ).on_close(context)


def test_executable_strategy_fails_when_target_is_below_one_lot() -> None:
    symbol = Symbol.parse("600001.XSHG")
    snapshot = _snapshot(_observation(str(symbol), "10"))
    bar = DailyBar(
        symbol,
        TRADE_DATE,
        Decimal("200"),
        Decimal("200"),
        Decimal("200"),
        Decimal("200"),
        Decimal("100000"),
        Decimal("20000000"),
    )
    context = StrategyContext(
        MarketSession(
            TRADE_DATE,
            datetime(2026, 7, 17, 1, 30, tzinfo=UTC),
            ASOF,
            (bar,),
        ),
        PortfolioSnapshot(
            ASOF,
            Decimal("10000"),
            Decimal("0"),
            Decimal("10000"),
            Decimal("0"),
            (),
        ),
    )
    strategy = MicrocapEqualWeightStrategy(
        snapshots={TRADE_DATE: snapshot},
        rebalance_dates={TRADE_DATE},
        universe=MicrocapUniverseConfig(target_count=1, minimum_constituents=1),
    )

    with pytest.raises(ValueError, match="below one board lot"):
        strategy.on_close(context)
