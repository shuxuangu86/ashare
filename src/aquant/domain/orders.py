import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from uuid import UUID, uuid4

from aquant.domain.enums import OrderStatus, Side
from aquant.domain.identifiers import Symbol
from aquant.domain.time import require_aware

_TERMINAL_ORDER_STATUSES = frozenset(
    {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}
)

_ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.DRAFT: frozenset({OrderStatus.PENDING_APPROVAL, OrderStatus.CANCELLED}),
    OrderStatus.PENDING_APPROVAL: frozenset(
        {OrderStatus.APPROVED, OrderStatus.REJECTED, OrderStatus.CANCELLED}
    ),
    OrderStatus.APPROVED: frozenset({OrderStatus.SUBMITTING, OrderStatus.CANCELLED}),
    OrderStatus.SUBMITTING: frozenset(
        {OrderStatus.SUBMITTED, OrderStatus.REJECTED, OrderStatus.UNKNOWN}
    ),
    OrderStatus.SUBMITTED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCEL_PENDING,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.UNKNOWN,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {
            OrderStatus.FILLED,
            OrderStatus.CANCEL_PENDING,
            OrderStatus.CANCELLED,
            OrderStatus.UNKNOWN,
        }
    ),
    OrderStatus.CANCEL_PENDING: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.UNKNOWN,
        }
    ),
    OrderStatus.UNKNOWN: frozenset(
        {
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        }
    ),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
}


def build_idempotency_key(
    *,
    account_id: str,
    strategy_id: str,
    trade_date: date,
    symbol: Symbol,
    side: Side,
    target_quantity: int,
) -> str:
    """Build a stable, versioned key for one logical order intent."""
    payload = {
        "account_id": account_id,
        "schema": "aquant.order-intent.v1",
        "side": side.value,
        "strategy_id": strategy_id,
        "symbol": symbol.canonical,
        "target_quantity": target_quantity,
        "trade_date": trade_date.isoformat(),
    }
    canonical = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class OrderIntent:
    account_id: str
    strategy_id: str
    trade_date: date
    symbol: Symbol
    side: Side
    target_quantity: int
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    intent_id: UUID = field(default_factory=uuid4)
    status: OrderStatus = OrderStatus.DRAFT
    idempotency_key: str = field(init=False)

    def __post_init__(self) -> None:
        account_id = self.account_id.strip()
        strategy_id = self.strategy_id.strip()
        if not account_id:
            raise ValueError("account_id must not be blank")
        if not strategy_id:
            raise ValueError("strategy_id must not be blank")
        if isinstance(self.target_quantity, bool) or self.target_quantity < 0:
            raise ValueError("target_quantity must be a non-negative integer")
        if not isinstance(self.target_quantity, int):
            raise TypeError("target_quantity must be an integer")

        created_at = require_aware(self.created_at, field_name="created_at")
        object.__setattr__(self, "account_id", account_id)
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(
            self,
            "idempotency_key",
            build_idempotency_key(
                account_id=account_id,
                strategy_id=strategy_id,
                trade_date=self.trade_date,
                symbol=self.symbol,
                side=self.side,
                target_quantity=self.target_quantity,
            ),
        )


def validate_order_transition(current: OrderStatus, target: OrderStatus) -> None:
    """Reject impossible or unsafe order-state transitions."""
    if current in _TERMINAL_ORDER_STATUSES:
        raise ValueError(f"terminal order status {current} cannot transition to {target}")
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"invalid order status transition: {current} -> {target}")
