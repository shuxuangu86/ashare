from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest

from aquant.backtest.accounting import PortfolioLedger, Position
from aquant.backtest.matching import (
    BacktestOrder,
    BacktestOrderStatus,
    Fill,
    FillEvent,
    OrderBook,
    OrderRequest,
    OrderSubmittedEvent,
)
from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol

SYMBOL = Symbol.parse("600000.XSHG")
SUBMITTED = datetime(2026, 7, 16, 7, tzinfo=UTC)
ORDER_ID = UUID("00000000-0000-0000-0000-000000000001")


def _order(**overrides: object) -> BacktestOrder:
    values: dict[str, object] = {
        "order_id": ORDER_ID,
        "symbol": SYMBOL,
        "side": Side.BUY,
        "quantity": 1000,
        "submitted_at": SUBMITTED,
    }
    values.update(overrides)
    return BacktestOrder(**values)  # type: ignore[arg-type]


def _fill(sequence: int, **overrides: object) -> Fill:
    values: dict[str, object] = {
        "fill_id": UUID(f"00000000-0000-0000-0000-{sequence:012d}"),
        "order_id": ORDER_ID,
        "symbol": SYMBOL,
        "side": Side.BUY,
        "quantity": 1000,
        "price": Decimal("10"),
        "fee": Decimal("5"),
        "occurred_at": SUBMITTED + timedelta(days=1),
    }
    values.update(overrides)
    return Fill(**values)  # type: ignore[arg-type]


def test_order_book_supports_partial_then_full_fill() -> None:
    book = OrderBook()
    book.submit(_order())
    assert book.eligible_orders(SUBMITTED) == ()
    assert book.eligible_orders(SUBMITTED + timedelta(seconds=1))[0].remaining_quantity == 1000

    first = _fill(101, quantity=400)
    partial = book.apply_fill(first)
    assert partial.status is BacktestOrderStatus.PARTIALLY_FILLED
    assert partial.remaining_quantity == 600

    second = _fill(102, quantity=600, occurred_at=first.occurred_at + timedelta(seconds=1))
    completed = book.apply_fill(second)
    assert completed.status is BacktestOrderStatus.FILLED
    assert completed.remaining_quantity == 0
    assert book.eligible_orders(second.occurred_at + timedelta(seconds=1)) == ()


def test_order_book_rejects_duplicates_mismatch_overfill_and_same_time_fill() -> None:
    book = OrderBook()
    book.submit(_order())
    with pytest.raises(ValueError, match="duplicate"):
        book.submit(_order())
    with pytest.raises(ValueError, match="strictly after"):
        book.apply_fill(_fill(201, occurred_at=SUBMITTED))
    with pytest.raises(ValueError, match="identity"):
        book.apply_fill(_fill(202, side=Side.SELL))
    with pytest.raises(ValueError, match="exceeds"):
        book.apply_fill(_fill(203, quantity=1001))
    with pytest.raises(KeyError, match="unknown"):
        book.apply_fill(_fill(204, order_id=UUID(int=999)))


def test_order_cancellation_and_events() -> None:
    book = OrderBook()
    order = _order()
    book.submit(order)
    cancelled = book.cancel(order.order_id)
    assert cancelled.status is BacktestOrderStatus.CANCELLED
    with pytest.raises(ValueError, match="open"):
        book.cancel(order.order_id)
    with pytest.raises(KeyError, match="unknown"):
        book.cancel(UUID(int=999))
    assert OrderSubmittedEvent(order).occurred_at == SUBMITTED
    fill = _fill(301)
    assert FillEvent(fill).occurred_at == fill.occurred_at


@pytest.mark.parametrize("quantity", [0, -1])
def test_order_request_requires_positive_integer(quantity: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        OrderRequest(SYMBOL, Side.BUY, quantity)
    with pytest.raises(TypeError, match="integer"):
        OrderRequest(SYMBOL, Side.BUY, True)


def test_fill_validates_price_fee_and_time() -> None:
    assert _fill(401).notional == Decimal("10000")
    with pytest.raises(ValueError, match="price"):
        _fill(402, price=Decimal("0"))
    with pytest.raises(ValueError, match="fee"):
        _fill(403, fee=Decimal("-1"))
    with pytest.raises(ValueError, match="timezone"):
        _fill(404, occurred_at=datetime(2026, 7, 17))


def test_ledger_buy_sell_snapshot_and_rebuild_are_exact() -> None:
    ledger = PortfolioLedger(Decimal("20000"))
    buy = _fill(501)
    ledger.apply_fill(buy)
    assert ledger.cash == Decimal("9995")

    sell = _fill(
        502,
        order_id=UUID(int=2),
        side=Side.SELL,
        quantity=400,
        price=Decimal("12"),
        fee=Decimal("3"),
        occurred_at=buy.occurred_at + timedelta(days=1),
    )
    ledger.apply_fill(sell)
    snapshot = ledger.snapshot(
        asof_time=sell.occurred_at,
        prices={SYMBOL: Decimal("11")},
    )

    assert snapshot.cash == Decimal("14792")
    assert snapshot.positions == (Position(SYMBOL, 600, Decimal("10.005")),)
    assert snapshot.market_value == Decimal("6600")
    assert snapshot.equity == Decimal("21392")
    assert snapshot.realized_pnl == Decimal("795.000")

    rebuilt = PortfolioLedger.rebuild(Decimal("20000"), ledger.fills)
    assert rebuilt.state_hash == ledger.state_hash
    assert rebuilt.snapshot(asof_time=sell.occurred_at, prices={SYMBOL: Decimal("11")}) == snapshot


def test_ledger_rejects_unfunded_oversold_duplicate_and_out_of_order_fills() -> None:
    ledger = PortfolioLedger(Decimal("100"))
    with pytest.raises(ValueError, match="cash"):
        ledger.apply_fill(_fill(601))

    funded = PortfolioLedger(Decimal("20000"))
    buy = _fill(602)
    funded.apply_fill(buy)
    with pytest.raises(ValueError, match="duplicate"):
        funded.apply_fill(buy)
    with pytest.raises(ValueError, match="available"):
        funded.apply_fill(_fill(603, side=Side.SELL, quantity=1001, occurred_at=buy.occurred_at))
    with pytest.raises(ValueError, match="chronological"):
        funded.apply_fill(_fill(604, occurred_at=buy.occurred_at - timedelta(seconds=1)))


def test_snapshot_requires_valid_price_for_every_position() -> None:
    ledger = PortfolioLedger(Decimal("20000"))
    ledger.apply_fill(_fill(701))
    with pytest.raises(KeyError, match="missing"):
        ledger.snapshot(asof_time=SUBMITTED + timedelta(days=1), prices={})
    with pytest.raises(ValueError, match="positive"):
        ledger.snapshot(asof_time=SUBMITTED + timedelta(days=1), prices={SYMBOL: Decimal("0")})
    with pytest.raises(ValueError, match="initial"):
        PortfolioLedger(Decimal("-1"))
