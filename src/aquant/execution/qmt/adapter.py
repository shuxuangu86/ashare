from dataclasses import dataclass, field
from typing import Protocol

from aquant.execution.broker_api import (
    BrokerAccountSnapshot,
    BrokerFill,
    BrokerOrder,
    BrokerOrderRequest,
)


class QmtGateway(Protocol):
    """Windows-side translation boundary implemented against the installed XtQuant version."""

    def submit_order(self, request: BrokerOrderRequest) -> BrokerOrder: ...

    def query_order(self, broker_order_id: str) -> BrokerOrder: ...

    def query_account(self) -> BrokerAccountSnapshot: ...

    def query_fills(self) -> tuple[BrokerFill, ...]: ...

    def cancel_order(self, broker_order_id: str) -> BrokerOrder: ...


@dataclass(slots=True)
class QmtAdapterState:
    idempotency_index: dict[str, str] = field(default_factory=dict)
    orders: dict[str, BrokerOrder] = field(default_factory=dict)


class QmtBrokerAdapter:
    """Version-neutral QMT adapter; the gateway is the only XtQuant-dependent component."""

    def __init__(self, gateway: QmtGateway, state: QmtAdapterState) -> None:
        self._gateway = gateway
        self._state = state

    def submit_order(self, request: BrokerOrderRequest) -> BrokerOrder:
        existing = self._state.idempotency_index.get(request.idempotency_key)
        if existing is not None:
            return self.query_order(existing)
        order = self._gateway.submit_order(request)
        self._state.idempotency_index[request.idempotency_key] = order.broker_order_id
        self._state.orders[order.broker_order_id] = order
        return order

    def query_order(self, broker_order_id: str) -> BrokerOrder:
        order = self._gateway.query_order(broker_order_id)
        self._state.orders[broker_order_id] = order
        return order

    def query_account(self) -> BrokerAccountSnapshot:
        return self._gateway.query_account()

    def query_fills(self) -> tuple[BrokerFill, ...]:
        return self._gateway.query_fills()

    def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        order = self._gateway.cancel_order(broker_order_id)
        self._state.orders[broker_order_id] = order
        return order
