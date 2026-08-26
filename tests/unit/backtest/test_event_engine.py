from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest

from aquant.backtest import (
    BacktestOrder,
    BacktestOrderStatus,
    EventDrivenBacktest,
    MarketSession,
    NextOpenMatcher,
    OrderRequest,
    StrategyContext,
)
from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar

SYMBOL = Symbol.parse("600000.XSHG")


def _bar(trade_date: date, *, open_price: str, close: str) -> DailyBar:
    opening = Decimal(open_price)
    closing = Decimal(close)
    return DailyBar(
        SYMBOL,
        trade_date,
        opening,
        max(opening, closing),
        min(opening, closing),
        closing,
        Decimal("1000000"),
        Decimal("10000000"),
    )


def _session(day: int, *, open_price: str, close: str) -> MarketSession:
    trade_date = date(2026, 7, day)
    return MarketSession(
        trade_date,
        datetime(2026, 7, day, 1, 30, tzinfo=UTC),
        datetime(2026, 7, day, 7, tzinfo=UTC),
        (_bar(trade_date, open_price=open_price, close=close),),
    )


SESSIONS = (
    _session(15, open_price="10", close="10"),
    _session(16, open_price="11", close="12"),
    _session(17, open_price="13", close="13"),
)


class BuyThenSell:
    def on_close(self, context: StrategyContext) -> tuple[OrderRequest, ...]:
        if context.session.trade_date.day == 15:
            return (OrderRequest(SYMBOL, Side.BUY, 100),)
        if context.session.trade_date.day == 16:
            return (OrderRequest(SYMBOL, Side.SELL, 100),)
        return ()


def test_end_to_end_next_open_backtest_and_accounting() -> None:
    result = EventDrivenBacktest(run_id="golden-day9", initial_cash=Decimal("10000")).run(
        SESSIONS, BuyThenSell()
    )

    assert len(result.orders) == 2
    assert all(order.status is BacktestOrderStatus.FILLED for order in result.orders)
    assert [fill.price for fill in result.fills] == [Decimal("11"), Decimal("13")]
    assert all(
        fill.occurred_at > order.submitted_at
        for fill, order in zip(result.fills, result.orders, strict=True)
    )
    assert [snapshot.equity for snapshot in result.equity_curve] == [
        Decimal("10000"),
        Decimal("10100"),
        Decimal("10200"),
    ]
    assert result.equity_curve[-1].cash == Decimal("10200")
    assert result.equity_curve[-1].positions == ()
    assert len(result.final_state_hash) == 64


def test_missing_bar_carries_forward_last_mark_without_creating_a_fill() -> None:
    other = Symbol.parse("000001.XSHE")
    day17 = date(2026, 7, 17)
    other_bar = DailyBar(
        other,
        day17,
        Decimal("5"),
        Decimal("5"),
        Decimal("5"),
        Decimal("5"),
        Decimal("100"),
        Decimal("500"),
    )
    sessions = (
        SESSIONS[0],
        SESSIONS[1],
        MarketSession(
            day17,
            datetime(2026, 7, 17, 1, 30, tzinfo=UTC),
            datetime(2026, 7, 17, 7, tzinfo=UTC),
            (other_bar,),
        ),
    )

    result = EventDrivenBacktest(run_id="stale-mark", initial_cash=Decimal("10000")).run(
        sessions, BuyThenSell()
    )

    assert [snapshot.equity for snapshot in result.equity_curve] == [
        Decimal("10000"),
        Decimal("10100"),
        Decimal("10100"),
    ]
    assert len(result.fills) == 1


def test_same_inputs_produce_identical_orders_fills_and_result_hash() -> None:
    first = EventDrivenBacktest(run_id="repeatable", initial_cash=Decimal("10000")).run(
        SESSIONS, BuyThenSell()
    )
    second = EventDrivenBacktest(run_id="repeatable", initial_cash=Decimal("10000")).run(
        SESSIONS, BuyThenSell()
    )
    assert first.orders == second.orders
    assert first.fills == second.fills
    assert first.final_state_hash == second.final_state_hash


def test_checkpoint_preserves_boundary_order_cash_positions_and_marks() -> None:
    continuous = EventDrivenBacktest(run_id="checkpoint-part-1", initial_cash=Decimal("10000")).run(
        SESSIONS, BuyThenSell()
    )
    first = EventDrivenBacktest(run_id="checkpoint-part-1", initial_cash=Decimal("10000")).run(
        SESSIONS[:1], BuyThenSell()
    )
    assert first.checkpoint is not None
    assert len(first.checkpoint.open_orders) == 1
    second = EventDrivenBacktest(
        run_id="checkpoint-part-2",
        initial_cash=first.equity_curve[-1].equity,
    ).run(SESSIONS[1:], BuyThenSell(), checkpoint=first.checkpoint)

    assert [fill.price for fill in second.fills] == [Decimal("11"), Decimal("13")]
    assert second.equity_curve == continuous.equity_curve[1:]
    assert second.checkpoint is not None
    assert second.checkpoint.ledger_state.cash == continuous.checkpoint.ledger_state.cash
    assert second.checkpoint.ledger_state.positions == continuous.checkpoint.ledger_state.positions


class UnfundedBuyer:
    def on_close(self, context: StrategyContext) -> tuple[OrderRequest, ...]:
        return (
            (OrderRequest(SYMBOL, Side.BUY, 200),) if context.session.trade_date.day == 15 else ()
        )


def test_cash_constraint_caps_fill_and_leaves_remainder_open() -> None:
    result = EventDrivenBacktest(run_id="cash-cap", initial_cash=Decimal("1100")).run(
        SESSIONS, UnfundedBuyer()
    )
    assert len(result.fills) == 1
    assert result.fills[0].quantity == 100
    assert result.orders[0].status is BacktestOrderStatus.PARTIALLY_FILLED
    assert result.orders[0].remaining_quantity == 100
    assert result.equity_curve[-1].positions[0].quantity == 100


def test_next_open_matcher_enforces_symbol_time_and_quantity_cap() -> None:
    submitted = datetime(2026, 7, 15, 7, tzinfo=UTC)
    order = BacktestOrder(UUID(int=1), SYMBOL, Side.BUY, 100, submitted)
    bar = SESSIONS[1].bars[0]
    matcher = NextOpenMatcher()
    fill = matcher.match(
        order,
        bar,
        occurred_at=SESSIONS[1].open_at,
        fill_id=UUID(int=2),
        maximum_quantity=40,
    )
    assert fill is not None and fill.quantity == 40 and fill.price == Decimal("11")
    assert (
        matcher.match(
            order,
            bar,
            occurred_at=SESSIONS[1].open_at,
            fill_id=UUID(int=3),
            maximum_quantity=0,
        )
        is None
    )
    with pytest.raises(ValueError, match="negative"):
        matcher.match(
            order,
            bar,
            occurred_at=SESSIONS[1].open_at,
            fill_id=UUID(int=4),
            maximum_quantity=-1,
        )
    with pytest.raises(ValueError, match="after"):
        matcher.match(order, bar, occurred_at=submitted, fill_id=UUID(int=5))


def test_market_sessions_and_engine_reject_invalid_timelines() -> None:
    with pytest.raises(ValueError, match="requires"):
        EventDrivenBacktest(run_id="empty", initial_cash=Decimal("1")).run((), BuyThenSell())
    with pytest.raises(ValueError, match="blank"):
        EventDrivenBacktest(run_id=" ", initial_cash=Decimal("1"))
    with pytest.raises(ValueError, match="chronological"):
        EventDrivenBacktest(run_id="bad", initial_cash=Decimal("1")).run(
            (SESSIONS[1], SESSIONS[0]), BuyThenSell()
        )
    session = SESSIONS[0]
    with pytest.raises(ValueError, match="ordered"):
        MarketSession(
            session.trade_date,
            session.close_at,
            session.open_at,
            session.bars,
        )
    wrong_day = session.trade_date + timedelta(days=1)
    with pytest.raises(ValueError, match="date mismatch"):
        MarketSession(
            session.trade_date,
            session.open_at,
            session.close_at,
            (_bar(wrong_day, open_price="10", close="10"),),
        )
