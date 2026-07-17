from typing import Protocol, runtime_checkable

from aquant.execution.broker_api.models import (
    BrokerAccountSnapshot,
    BrokerFill,
    BrokerOrder,
    BrokerOrderRequest,
)


@runtime_checkable
class BrokerAdapter(Protocol):
    def submit_order(self, request: BrokerOrderRequest) -> BrokerOrder: ...

    def query_order(self, broker_order_id: str) -> BrokerOrder: ...

    def query_account(self) -> BrokerAccountSnapshot: ...

    def query_fills(self) -> tuple[BrokerFill, ...]: ...

    def cancel_order(self, broker_order_id: str) -> BrokerOrder: ...
