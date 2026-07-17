from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal

from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.execution.broker_api import (
    BrokerAccountSnapshot,
    BrokerFill,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderStatus,
)


@dataclass(slots=True)
class PaperBrokerState:
    cash: Decimal
    positions: dict[Symbol, int] = field(default_factory=dict)
    prices: dict[Symbol, Decimal] = field(default_factory=dict)
    orders: dict[str, BrokerOrder] = field(default_factory=dict)
    idempotency_index: dict[str, str] = field(default_factory=dict)
    fills: list[BrokerFill] = field(default_factory=list)
    sequence: int = 0


class PaperBroker:
    """Synchronous paper broker whose shared state survives agent reconstruction."""

    def __init__(self, state: PaperBrokerState) -> None:
        if state.cash < 0:
            raise ValueError("paper broker cash cannot be negative")
        self._state = state

    def set_price(self, symbol: Symbol, price: Decimal) -> None:
        resolved = Decimal(price)
        if resolved <= 0:
            raise ValueError("paper broker price must be positive")
        self._state.prices[symbol] = resolved

    def submit_order(self, request: BrokerOrderRequest) -> BrokerOrder:
        existing_id = self._state.idempotency_index.get(request.idempotency_key)
        if existing_id is not None:
            return self._state.orders[existing_id]
        self._state.sequence += 1
        order_id = f"PAPER-O-{self._state.sequence:08d}"
        now = datetime.now(UTC)
        order = BrokerOrder(order_id, request, BrokerOrderStatus.SUBMITTED, now)
        self._state.orders[order_id] = order
        self._state.idempotency_index[request.idempotency_key] = order_id
        price = self._state.prices.get(request.symbol)
        if price is None:
            return order
        notional = price * request.quantity
        if request.side is Side.BUY and notional > self._state.cash:
            rejected = replace(order, status=BrokerOrderStatus.REJECTED)
            self._state.orders[order_id] = rejected
            return rejected
        if request.side is Side.SELL and request.quantity > self._state.positions.get(
            request.symbol, 0
        ):
            rejected = replace(order, status=BrokerOrderStatus.REJECTED)
            self._state.orders[order_id] = rejected
            return rejected
        if request.side is Side.BUY:
            self._state.cash -= notional
            self._state.positions[request.symbol] = (
                self._state.positions.get(request.symbol, 0) + request.quantity
            )
        else:
            self._state.cash += notional
            self._state.positions[request.symbol] -= request.quantity
        trade_id = f"PAPER-T-{self._state.sequence:08d}"
        self._state.fills.append(
            BrokerFill(
                trade_id,
                order_id,
                request.symbol,
                request.side,
                request.quantity,
                price,
                now,
            )
        )
        filled = replace(
            order,
            status=BrokerOrderStatus.FILLED,
            filled_quantity=request.quantity,
        )
        self._state.orders[order_id] = filled
        return filled

    def query_order(self, broker_order_id: str) -> BrokerOrder:
        return self._state.orders[broker_order_id]

    def query_account(self) -> BrokerAccountSnapshot:
        return BrokerAccountSnapshot(
            datetime.now(UTC),
            self._state.cash,
            dict(self._state.positions),
        )

    def query_fills(self) -> tuple[BrokerFill, ...]:
        return tuple(self._state.fills)

    def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        order = self._state.orders[broker_order_id]
        if order.status is not BrokerOrderStatus.SUBMITTED:
            raise ValueError("only submitted paper orders can be cancelled")
        cancelled = replace(order, status=BrokerOrderStatus.CANCELLED)
        self._state.orders[broker_order_id] = cancelled
        return cancelled
