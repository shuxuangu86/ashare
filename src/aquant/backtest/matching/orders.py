from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from aquant.backtest.event_engine.events import EventKind
from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.domain.time import require_aware


class BacktestOrderStatus(StrEnum):
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class OrderRequest:
    symbol: Symbol
    side: Side
    quantity: int

    def __post_init__(self) -> None:
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise TypeError("order quantity must be an integer")
        if self.quantity <= 0:
            raise ValueError("order quantity must be positive")


@dataclass(frozen=True, slots=True)
class BacktestOrder:
    order_id: UUID
    symbol: Symbol
    side: Side
    quantity: int
    submitted_at: datetime
    filled_quantity: int = 0
    status: BacktestOrderStatus = BacktestOrderStatus.OPEN

    def __post_init__(self) -> None:
        OrderRequest(self.symbol, self.side, self.quantity)
        if self.filled_quantity < 0 or self.filled_quantity > self.quantity:
            raise ValueError("filled quantity must be within order quantity")
        object.__setattr__(
            self,
            "submitted_at",
            require_aware(self.submitted_at, field_name="submitted_at"),
        )
        expected = (
            BacktestOrderStatus.FILLED
            if self.filled_quantity == self.quantity
            else BacktestOrderStatus.PARTIALLY_FILLED
            if self.filled_quantity > 0
            else BacktestOrderStatus.OPEN
        )
        if self.status is not BacktestOrderStatus.CANCELLED and self.status is not expected:
            raise ValueError("order status is inconsistent with filled quantity")

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: UUID
    order_id: UUID
    symbol: Symbol
    side: Side
    quantity: int
    price: Decimal
    fee: Decimal
    occurred_at: datetime

    def __post_init__(self) -> None:
        OrderRequest(self.symbol, self.side, self.quantity)
        price = Decimal(self.price)
        fee = Decimal(self.fee)
        if not price.is_finite() or price <= 0:
            raise ValueError("fill price must be finite and positive")
        if not fee.is_finite() or fee < 0:
            raise ValueError("fill fee must be finite and non-negative")
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "fee", fee)
        object.__setattr__(
            self,
            "occurred_at",
            require_aware(self.occurred_at, field_name="occurred_at"),
        )

    @property
    def notional(self) -> Decimal:
        return self.price * self.quantity


@dataclass(frozen=True, slots=True)
class OrderSubmittedEvent:
    order: BacktestOrder
    occurred_at: datetime = field(init=False)
    kind: EventKind = field(default=EventKind.ORDER_SUBMITTED, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "occurred_at", self.order.submitted_at)


@dataclass(frozen=True, slots=True)
class FillEvent:
    fill: Fill
    occurred_at: datetime = field(init=False)
    kind: EventKind = field(default=EventKind.FILL, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "occurred_at", self.fill.occurred_at)


class OrderBook:
    def __init__(self) -> None:
        self._orders: dict[UUID, BacktestOrder] = {}
        self._fill_ids: set[UUID] = set()

    def submit(self, order: BacktestOrder) -> None:
        if order.order_id in self._orders:
            raise ValueError(f"duplicate backtest order id: {order.order_id}")
        self._orders[order.order_id] = order

    def eligible_orders(self, occurred_at: datetime) -> tuple[BacktestOrder, ...]:
        timestamp = require_aware(occurred_at, field_name="occurred_at")
        return tuple(
            order
            for order in self._orders.values()
            if order.status in {BacktestOrderStatus.OPEN, BacktestOrderStatus.PARTIALLY_FILLED}
            and order.submitted_at < timestamp
        )

    def validate_fill(self, fill: Fill) -> BacktestOrder:
        if fill.fill_id in self._fill_ids:
            raise ValueError(f"duplicate fill id: {fill.fill_id}")
        try:
            order = self._orders[fill.order_id]
        except KeyError as exc:
            raise KeyError(f"fill references unknown order: {fill.order_id}") from exc
        if order.status not in {BacktestOrderStatus.OPEN, BacktestOrderStatus.PARTIALLY_FILLED}:
            raise ValueError("cannot fill a terminal backtest order")
        if fill.symbol != order.symbol or fill.side is not order.side:
            raise ValueError("fill identity does not match its order")
        if fill.quantity > order.remaining_quantity:
            raise ValueError("fill quantity exceeds remaining order quantity")
        if fill.occurred_at <= order.submitted_at:
            raise ValueError("fill must occur strictly after order submission")
        return order

    def apply_fill(self, fill: Fill) -> BacktestOrder:
        order = self.validate_fill(fill)
        filled_quantity = order.filled_quantity + fill.quantity
        status = (
            BacktestOrderStatus.FILLED
            if filled_quantity == order.quantity
            else BacktestOrderStatus.PARTIALLY_FILLED
        )
        updated = replace(order, filled_quantity=filled_quantity, status=status)
        self._orders[order.order_id] = updated
        self._fill_ids.add(fill.fill_id)
        return updated

    def cancel(self, order_id: UUID) -> BacktestOrder:
        try:
            order = self._orders[order_id]
        except KeyError as exc:
            raise KeyError(f"unknown backtest order: {order_id}") from exc
        if order.status not in {BacktestOrderStatus.OPEN, BacktestOrderStatus.PARTIALLY_FILLED}:
            raise ValueError("only open orders can be cancelled")
        updated = replace(order, status=BacktestOrderStatus.CANCELLED)
        self._orders[order_id] = updated
        return updated

    def get(self, order_id: UUID) -> BacktestOrder:
        return self._orders[order_id]

    @property
    def orders(self) -> tuple[BacktestOrder, ...]:
        return tuple(self._orders.values())
