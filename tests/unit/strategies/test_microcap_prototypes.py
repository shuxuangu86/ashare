from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from aquant.backtest import MarketSession, OrderRequest, StrategyContext
from aquant.backtest.accounting import PortfolioSnapshot, Position
from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar
from aquant.strategies.microcap import (
    MicrocapObservation,
    MicrocapPrototype,
    MicrocapPrototypeSelector,
    MicrocapPrototypeStrategy,
    MicrocapSnapshot,
)

TRADE_DATE = date(2026, 7, 17)
ASOF = datetime(2026, 7, 17, 11, 30, tzinfo=UTC)


def _observation(
    rank: int,
    *,
    pb: Decimal | None = None,
    growth: Decimal | None = None,
    dividend: Decimal | None = None,
    leverage: Decimal | None = None,
    turnover_volatility: Decimal | None = None,
    delisting_risk: bool = False,
) -> MicrocapObservation:
    value = Decimal(rank)
    return MicrocapObservation(
        Symbol.parse(f"{600000 + rank:06d}.XSHG"),
        TRADE_DATE,
        ASOF,
        date(2020, 1, 1),
        value,
        False,
        False,
        float_market_cap=value,
        pb=pb,
        net_profit_yoy=growth,
        dividend_yield=dividend,
        debt_to_assets=leverage,
        turnover_volatility_20d=turnover_volatility,
        is_delisting_risk=delisting_risk,
    )


def _snapshot(observations: tuple[MicrocapObservation, ...]) -> MicrocapSnapshot:
    return MicrocapSnapshot(TRADE_DATE, ASOF, observations)


def test_size_slices_are_disjoint_and_reconcile_to_smallest_400() -> None:
    observations = tuple(_observation(rank) for rank in range(1, 451))
    snapshot = _snapshot(observations)

    smallest_100 = MicrocapPrototypeSelector(MicrocapPrototype.SMALLEST_100).select(snapshot)
    rank_101_400 = MicrocapPrototypeSelector(MicrocapPrototype.RANK_101_400).select(snapshot)
    smallest_400 = MicrocapPrototypeSelector(MicrocapPrototype.SMALLEST_400).select(snapshot)

    assert len(smallest_100.symbols) == 100
    assert len(rank_101_400.symbols) == 300
    assert set(smallest_100.symbols).isdisjoint(rank_101_400.symbols)
    assert set(smallest_400.symbols) == set(smallest_100.symbols) | set(rank_101_400.symbols)


def test_executable_95_applies_growth_filter_and_120_rank_holding_buffer() -> None:
    observations = tuple(
        _observation(
            rank,
            pb=Decimal("1") if rank <= 130 else Decimal("-1"),
            growth=Decimal("0.1") if rank <= 130 else Decimal("-0.1"),
        )
        for rank in range(1, 451)
    )
    held_at_rank_110 = observations[109].symbol

    selection = MicrocapPrototypeSelector(MicrocapPrototype.EXECUTABLE_95).select(
        _snapshot(observations),
        held_symbols=(held_at_rank_110,),
    )

    assert len(selection.symbols) == 95
    assert held_at_rank_110 in selection.symbols
    assert observations[94].symbol not in selection.symbols
    assert dict(selection.excluded_counts)["factor_filter"] == 320


def test_dividend_quality_pipeline_is_sequential_and_uses_float_market_cap() -> None:
    observations = tuple(
        _observation(
            rank,
            dividend=Decimal(rank),
            leverage=Decimal(rank),
            turnover_volatility=Decimal(rank),
        )
        for rank in range(1, 451)
    )

    selection = MicrocapPrototypeSelector(MicrocapPrototype.DIVIDEND_QUALITY_10).select(
        _snapshot(observations)
    )

    assert selection.symbols == tuple(item.symbol for item in observations[225:235])


def test_low_pb_low_turnover_pipeline_selects_35_smallest_survivors() -> None:
    observations = tuple(
        _observation(
            rank,
            pb=Decimal(rank),
            turnover_volatility=Decimal(rank),
        )
        for rank in range(1, 451)
    )

    selection = MicrocapPrototypeSelector(MicrocapPrototype.LOW_PB_LOW_TURNOVER_35).select(
        _snapshot(observations)
    )

    assert selection.symbols == tuple(item.symbol for item in observations[:35])


def test_factor_prototypes_fail_closed_on_missing_fields_and_delisting_risk() -> None:
    observations = tuple(_observation(rank) for rank in range(1, 451))
    selector = MicrocapPrototypeSelector(MicrocapPrototype.DIVIDEND_QUALITY_10)
    with pytest.raises(ValueError, match="factor-eligible"):
        selector.select(_snapshot(observations))

    risky = _observation(451, delisting_risk=True)
    safe = tuple(
        _observation(
            rank,
            dividend=Decimal(rank),
            leverage=Decimal(rank),
            turnover_volatility=Decimal(rank),
        )
        for rank in range(1, 451)
    )
    selection = selector.select(_snapshot((*safe, risky)))
    assert risky.symbol not in selection.symbols
    assert dict(selection.excluded_counts)["delisting_risk"] == 1


def test_observation_validates_optional_factor_values() -> None:
    with pytest.raises(ValueError, match="float market cap"):
        MicrocapObservation(
            Symbol.parse("600001.XSHG"),
            TRADE_DATE,
            ASOF,
            date(2020, 1, 1),
            Decimal("1"),
            False,
            False,
            float_market_cap=Decimal("0"),
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        _observation(1, turnover_volatility=Decimal("-0.1"))


def test_non_rebalance_day_runs_regulatory_hard_exit_without_reselecting() -> None:
    risky = _observation(1, delisting_risk=True)
    bar = DailyBar(
        risky.symbol,
        TRADE_DATE,
        Decimal("10"),
        Decimal("10"),
        Decimal("10"),
        Decimal("10"),
        Decimal("100000"),
        Decimal("1000000"),
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
            Decimal("0"),
            Decimal("1000"),
            Decimal("1000"),
            Decimal("0"),
            (Position(risky.symbol, 100, Decimal("10")),),
        ),
    )
    strategy = MicrocapPrototypeStrategy(
        prototype=MicrocapPrototype.SMALLEST_100,
        snapshots={TRADE_DATE: _snapshot((risky,))},
        rebalance_dates=set(),
    )

    assert strategy.on_close(context) == (OrderRequest(risky.symbol, Side.SELL, 100),)
    with pytest.raises(KeyError, match="daily micro-cap risk"):
        MicrocapPrototypeStrategy(
            prototype=MicrocapPrototype.SMALLEST_100,
            snapshots={},
            rebalance_dates=set(),
        ).on_close(context)
