from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.domain.time import require_aware


class BrokerOrderStatus(StrEnum):
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class BrokerOrderRequest:
    client_order_id: str
    idempotency_key: str
    symbol: Symbol
    side: Side
    quantity: int

    def __post_init__(self) -> None:
        if not self.client_order_id.strip() or len(self.idempotency_key) != 64:
            raise ValueError("broker request ids are invalid")
        if (
            isinstance(self.quantity, bool)
            or not isinstance(self.quantity, int)
            or self.quantity <= 0
        ):
            raise ValueError("broker order quantity must be a positive integer")


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    broker_order_id: str
    request: BrokerOrderRequest
    status: BrokerOrderStatus
    submitted_at: datetime
    filled_quantity: int = 0

    def __post_init__(self) -> None:
        if (
            not self.broker_order_id.strip()
            or not 0 <= self.filled_quantity <= self.request.quantity
        ):
            raise ValueError("broker order state is invalid")
        object.__setattr__(
            self,
            "submitted_at",
            require_aware(self.submitted_at, field_name="submitted_at"),
        )


@dataclass(frozen=True, slots=True)
class BrokerFill:
    broker_trade_id: str
    broker_order_id: str
    symbol: Symbol
    side: Side
    quantity: int
    price: Decimal
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class BrokerAccountSnapshot:
    asof_time: datetime
    cash: Decimal
    positions: dict[Symbol, int]
