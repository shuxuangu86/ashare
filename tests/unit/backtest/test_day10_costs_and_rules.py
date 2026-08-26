from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest

from aquant.backtest import (
    AshareExecutionRules,
    AshareFeeModel,
    AshareFeeSchedule,
    AshareOpenMatcher,
    BacktestOrder,
    BacktestOrderStatus,
    EventDrivenBacktest,
    MarketSession,
    OrderRequest,
    StrategyContext,
    UnfilledOrderPolicy,
    calculate_metrics,
    render_markdown_report,
)
from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar, LimitStatus, SecurityStatus

SYMBOL = Symbol.parse("600000.XSHG")
SUBMITTED = datetime(2026, 7, 15, 7, tzinfo=UTC)
OPENED = SUBMITTED + timedelta(days=1)


def _order(side: Side = Side.BUY, quantity: int = 1000) -> BacktestOrder:
    return BacktestOrder(UUID(int=1), SYMBOL, side, quantity, SUBMITTED)


def _bar(volume: str = "10000") -> DailyBar:
    return DailyBar(
        SYMBOL,
        date(2026, 7, 16),
        Decimal("10"),
        Decimal("10.5"),
        Decimal("9.5"),
        Decimal("10"),
        Decimal(volume),
        Decimal("100000"),
    )


def _status(**overrides: object) -> SecurityStatus:
    values: dict[str, object] = {
        "symbol": SYMBOL,
        "trade_date": date(2026, 7, 16),
        "suspended": False,
        "is_st": False,
        "limit_status": LimitStatus.NONE,
    }
    values.update(overrides)
    return SecurityStatus(**values)  # type: ignore[arg-type]


def test_fee_model_applies_minimum_commission_and_sell_tax() -> None:
    model = AshareFeeModel()
    buy = model.calculate(Side.BUY, Decimal("1000"))
    sell = model.calculate(Side.SELL, Decimal("1000"))
    assert buy.commission == Decimal("5")
    assert buy.stamp_duty == 0
    assert sell.stamp_duty == Decimal("0.5000")
    assert sell.total > buy.total
    with pytest.raises(ValueError, match="positive"):
        model.calculate(Side.BUY, Decimal("0"))


def test_fee_schedule_uses_trade_date_and_fails_before_supported_history() -> None:
    schedule = AshareFeeSchedule()
    old = schedule.model_for(date(2021, 1, 4))
    transfer_reduced = schedule.model_for(date(2022, 4, 29))
    stamp_reduced = schedule.model_for(date(2023, 8, 28))

    assert old.transfer_fee_rate == Decimal("0.00002")
    assert transfer_reduced.transfer_fee_rate == Decimal("0.00001")
    assert transfer_reduced.sell_stamp_duty_rate == Decimal("0.001")
    assert stamp_reduced.sell_stamp_duty_rate == Decimal("0.0005")
    with pytest.raises(ValueError, match="before 2015"):
        schedule.model_for(date(2015, 7, 8))
    with pytest.raises(ValueError, match="either"):
        AshareOpenMatcher(fee_model=AshareFeeModel(), fee_schedule=schedule)


def test_rule_aware_matcher_applies_lot_volume_slippage_and_fees() -> None:
    matcher = AshareOpenMatcher(
        rules=AshareExecutionRules(
            lot_size=100,
            max_volume_participation=Decimal("0.05"),
            slippage_bps=Decimal("10"),
        )
    )
    fill = matcher.match(
        _order(quantity=1000),
        _bar(volume="9000"),
        occurred_at=OPENED,
        fill_id=UUID(int=2),
        available_cash=Decimal("100000"),
        status=_status(),
    )
    assert fill is not None
    assert fill.quantity == 400
    assert fill.price == Decimal("10.010")
    assert fill.fee > 0


@pytest.mark.parametrize(
    ("side", "expected"),
    [(Side.BUY, Decimal("10.5")), (Side.SELL, Decimal("9.5"))],
)
def test_slippage_price_is_clamped_to_observed_ohlc(side: Side, expected: Decimal) -> None:
    matcher = AshareOpenMatcher(rules=AshareExecutionRules(slippage_bps=Decimal("1000")))
    fill = matcher.match(
        _order(side=side),
        _bar(),
        occurred_at=OPENED,
        fill_id=UUID(int=22 if side is Side.BUY else 23),
        available_cash=Decimal("100000"),
        status=_status(),
    )
    assert fill is not None
    assert fill.price == expected


@pytest.mark.parametrize(
    ("side", "status"),
    [
        (Side.BUY, _status(suspended=True)),
        (Side.BUY, _status(limit_status=LimitStatus.LIMIT_UP)),
        (Side.SELL, _status(limit_status=LimitStatus.LIMIT_DOWN)),
    ],
)
def test_suspension_and_directional_limits_block_fills(side: Side, status: SecurityStatus) -> None:
    fill = AshareOpenMatcher().match(
        _order(side),
        _bar(),
        occurred_at=OPENED,
        fill_id=UUID(int=3),
        available_cash=Decimal("100000"),
        status=status,
    )
    assert fill is None


def test_missing_status_fails_closed_and_cash_includes_fees() -> None:
    matcher = AshareOpenMatcher()
    assert (
        matcher.match(
            _order(),
            _bar(),
            occurred_at=OPENED,
            fill_id=UUID(int=4),
            available_cash=Decimal("100000"),
        )
        is None
    )
    fill = matcher.match(
        _order(quantity=200),
        _bar(),
        occurred_at=OPENED,
        fill_id=UUID(int=5),
        available_cash=Decimal("1005"),
        status=_status(),
    )
    assert fill is None


def test_capacity_can_use_prior_20d_volume_without_seeing_fill_day_volume() -> None:
    matcher = AshareOpenMatcher(
        rules=AshareExecutionRules(
            max_volume_participation=Decimal("0.05"),
            use_prior_20d_average_volume=True,
        )
    )
    assert (
        matcher.match(
            _order(),
            _bar(volume="999999"),
            occurred_at=OPENED,
            fill_id=UUID(int=6),
            available_cash=Decimal("100000"),
            status=_status(),
        )
        is None
    )
    fill = matcher.match(
        _order(),
        _bar(volume="1"),
        occurred_at=OPENED,
        fill_id=UUID(int=7),
        available_cash=Decimal("100000"),
        status=_status(prior_20d_average_volume=Decimal("10000")),
    )
    assert fill is not None
    assert fill.quantity == 500


def test_rule_configuration_rejects_unsafe_values() -> None:
    with pytest.raises(ValueError):
        AshareExecutionRules(lot_size=0)
    with pytest.raises(ValueError):
        AshareExecutionRules(max_volume_participation=Decimal("1.1"))
    with pytest.raises(ValueError):
        AshareExecutionRules(slippage_bps=Decimal("-1"))
    with pytest.raises(ValueError):
        AshareFeeModel(commission_rate=Decimal("-1"))
    with pytest.raises(ValueError, match="average volume"):
        _status(prior_20d_average_volume=Decimal("-1"))


class BuyOnce:
    def on_close(self, context: StrategyContext) -> tuple[OrderRequest, ...]:
        return (
            (OrderRequest(SYMBOL, Side.BUY, 100),)
            if context.session.trade_date == date(2026, 7, 15)
            else ()
        )


def _session(day: int, *, with_status: bool) -> MarketSession:
    trade_date = date(2026, 7, day)
    bar = DailyBar(
        SYMBOL,
        trade_date,
        Decimal("10"),
        Decimal("10"),
        Decimal("10"),
        Decimal("10"),
        Decimal("10000"),
        Decimal("100000"),
    )
    statuses = (
        (SecurityStatus(SYMBOL, trade_date, False, False, LimitStatus.NONE),) if with_status else ()
    )
    return MarketSession(
        trade_date,
        datetime(2026, 7, day, 1, 30, tzinfo=UTC),
        datetime(2026, 7, day, 7, tzinfo=UTC),
        (bar,),
        statuses,
    )


def test_rule_aware_matcher_integrates_with_engine_and_report() -> None:
    matcher = AshareOpenMatcher(rules=AshareExecutionRules(slippage_bps=Decimal("0")))
    result = EventDrivenBacktest(
        run_id="day10-costs",
        initial_cash=Decimal("2000"),
        matcher=matcher,
    ).run((_session(15, with_status=True), _session(16, with_status=True)), BuyOnce())

    assert len(result.fills) == 1 and result.fills[0].fee == Decimal("5.01000")
    assert result.equity_curve[-1].equity == Decimal("1994.99000")
    metrics = calculate_metrics(result, initial_equity=Decimal("2000"))
    assert metrics.total_fees == Decimal("5.01000")
    assert metrics.fill_rate == Decimal("1")
    assert metrics.total_return < 0
    report = render_markdown_report(result, initial_equity=Decimal("2000"))
    assert "Backtest Report: day10-costs" in report
    assert "Total fees: 5.01000" in report


def test_rule_aware_engine_fails_closed_without_status() -> None:
    result = EventDrivenBacktest(
        run_id="day10-no-status",
        initial_cash=Decimal("2000"),
        matcher=AshareOpenMatcher(),
    ).run((_session(15, with_status=False), _session(16, with_status=False)), BuyOnce())
    assert result.fills == ()
    assert result.orders[0].filled_quantity == 0


def test_day_order_policy_prevents_a_blocked_order_from_filling_later() -> None:
    blocked = _session(16, with_status=True)
    blocked = MarketSession(
        blocked.trade_date,
        blocked.open_at,
        blocked.close_at,
        blocked.bars,
        (_status(limit_status=LimitStatus.LIMIT_UP),),
    )
    result = EventDrivenBacktest(
        run_id="day-order-cancel",
        initial_cash=Decimal("2000"),
        matcher=AshareOpenMatcher(),
        unfilled_order_policy=UnfilledOrderPolicy.CANCEL_AFTER_OPEN,
    ).run((_session(15, with_status=True), blocked, _session(17, with_status=True)), BuyOnce())

    assert result.fills == ()
    assert result.orders[0].status is BacktestOrderStatus.CANCELLED
