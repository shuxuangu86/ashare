from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from aquant.backtest import MarketSession, StrategyContext
from aquant.backtest.accounting import PortfolioSnapshot, Position
from aquant.domain.corporate_actions import CorporateAction, CorporateActionKind
from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar, LimitStatus, SecurityStatus
from aquant.strategies.microcap import MicrocapRankBandStrategy, MicrocapRankBandTarget

TRADE_DATE = date(2026, 7, 17)
ASOF = datetime(2026, 7, 17, 7, tzinfo=UTC)
CHEAP = Symbol.parse("600001.XSHG")
EXPENSIVE = Symbol.parse("600002.XSHG")
REMOVED = Symbol.parse("600003.XSHG")


def _bar(
    symbol: Symbol,
    price: str,
    *,
    trade_date: date = TRADE_DATE,
) -> DailyBar:
    value = Decimal(price)
    return DailyBar(
        symbol,
        trade_date,
        value,
        value,
        value,
        value,
        Decimal("1000000"),
        Decimal("10000000"),
    )


def _context(
    *,
    positions: tuple[Position, ...] = (),
    bars: tuple[DailyBar, ...] = (),
    statuses: tuple[SecurityStatus, ...] = (),
    equity: str = "10000",
    trade_date: date = TRADE_DATE,
    actions: tuple[CorporateAction, ...] = (),
) -> StrategyContext:
    asof = datetime(
        trade_date.year,
        trade_date.month,
        trade_date.day,
        7,
        tzinfo=UTC,
    )
    market_value = sum(
        (
            next(
                (bar.close for bar in bars if bar.symbol == position.symbol),
                position.average_cost,
            )
            * position.quantity
            for position in positions
        ),
        Decimal("0"),
    )
    return StrategyContext(
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
            actions,
        ),
        PortfolioSnapshot(
            asof,
            Decimal(equity) - market_value,
            market_value,
            Decimal(equity),
            Decimal("0"),
            positions,
        ),
    )


def test_rank_target_validates_interval_completeness_and_uniqueness() -> None:
    with pytest.raises(ValueError, match="interval"):
        MicrocapRankBandTarget(TRADE_DATE, 0, 1, (CHEAP,))
    with pytest.raises(ValueError, match="incomplete"):
        MicrocapRankBandTarget(TRADE_DATE, 1, 2, (CHEAP,))
    with pytest.raises(ValueError, match="duplicate"):
        MicrocapRankBandTarget(TRADE_DATE, 1, 2, (CHEAP, CHEAP))


def test_strategy_equal_weights_in_lots_and_leaves_unbuyable_target_as_cash() -> None:
    target = MicrocapRankBandTarget(TRADE_DATE, 1, 2, (CHEAP, EXPENSIVE))
    strategy = MicrocapRankBandStrategy(targets={TRADE_DATE: target})

    orders = strategy.on_close(_context(bars=(_bar(CHEAP, "10"), _bar(EXPENSIVE, "100"))))

    assert len(orders) == 1
    assert orders[0].symbol == CHEAP
    assert orders[0].side is Side.BUY
    assert orders[0].quantity == 400
    assert strategy.diagnostics[0].zero_lot_count == 1


def test_strategy_sells_removed_before_buy_and_retains_selected_without_bar() -> None:
    target = MicrocapRankBandTarget(TRADE_DATE, 1, 2, (CHEAP, EXPENSIVE))
    strategy = MicrocapRankBandStrategy(targets={TRADE_DATE: target})
    positions = (
        Position(EXPENSIVE, 100, Decimal("10")),
        Position(REMOVED, 100, Decimal("10")),
    )

    orders = strategy.on_close(
        _context(
            positions=positions,
            bars=(_bar(CHEAP, "10"), _bar(REMOVED, "10")),
        )
    )

    assert orders[0].symbol == REMOVED and orders[0].side is Side.SELL
    assert orders[1].symbol == CHEAP and orders[1].side is Side.BUY
    assert all(order.symbol != EXPENSIVE for order in orders)
    assert strategy.diagnostics[0].retained_without_bar_count == 1


def test_strategy_retains_holdings_inside_sell_buffer() -> None:
    target = MicrocapRankBandTarget(
        TRADE_DATE,
        1,
        3,
        (CHEAP, EXPENSIVE, REMOVED),
        target_count=2,
        sell_buffer_rank=3,
    )
    strategy = MicrocapRankBandStrategy(targets={TRADE_DATE: target})
    positions = (Position(REMOVED, 100, Decimal("10")),)

    orders = strategy.on_close(
        _context(
            positions=positions,
            bars=(
                _bar(CHEAP, "10"),
                _bar(EXPENSIVE, "10"),
                _bar(REMOVED, "10"),
            ),
        )
    )

    assert all(not (order.symbol == REMOVED and order.side is Side.SELL) for order in orders)
    assert {order.symbol for order in orders if order.side is Side.BUY} == {
        CHEAP,
        REMOVED,
    }


def test_non_rebalance_day_submits_only_visible_st_hard_exit() -> None:
    prior = date(2026, 7, 16)
    target = MicrocapRankBandTarget(prior, 1, 1, (CHEAP,))
    strategy = MicrocapRankBandStrategy(targets={prior: target})
    status = SecurityStatus(
        CHEAP,
        TRADE_DATE,
        False,
        True,
        LimitStatus.NONE,
        Decimal("1000000"),
    )
    position = Position(CHEAP, 100, Decimal("10"))

    orders = strategy.on_close(
        _context(positions=(position,), bars=(_bar(CHEAP, "10"),), statuses=(status,))
    )

    assert len(orders) == 1
    assert orders[0].side is Side.SELL


def test_unfilled_target_is_retried_without_changing_membership() -> None:
    prior = date(2026, 7, 16)
    target = MicrocapRankBandTarget(prior, 1, 1, (CHEAP,))
    strategy = MicrocapRankBandStrategy(targets={prior: target})
    prior_bar = _bar(CHEAP, "10", trade_date=prior)

    first_orders = strategy.on_close(_context(bars=(prior_bar,), trade_date=prior))
    retry_orders = strategy.on_close(_context(bars=(_bar(CHEAP, "12"),), trade_date=TRADE_DATE))

    assert first_orders[0].quantity == 900
    assert retry_orders == first_orders


def test_retry_target_tracks_stock_dividend_quantity_adjustment() -> None:
    prior = date(2026, 7, 16)
    target = MicrocapRankBandTarget(prior, 1, 1, (CHEAP,))
    strategy = MicrocapRankBandStrategy(targets={prior: target})
    strategy.on_close(
        _context(
            bars=(_bar(CHEAP, "10", trade_date=prior),),
            trade_date=prior,
        )
    )
    action = CorporateAction(
        uuid4(),
        CHEAP,
        CorporateActionKind.STOCK_DIVIDEND,
        datetime(2026, 7, 17, 1, 30, tzinfo=UTC),
        ratio=Decimal("1"),
    )

    orders = strategy.on_close(
        _context(
            positions=(Position(CHEAP, 1800, Decimal("5")),),
            bars=(_bar(CHEAP, "5"),),
            actions=(action,),
        )
    )

    assert orders == ()
