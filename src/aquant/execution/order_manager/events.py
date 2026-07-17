from dataclasses import dataclass
from datetime import datetime

from aquant.domain.time import require_aware
from aquant.execution.broker_api import BrokerOrderStatus

_TERMINAL = frozenset(
    {BrokerOrderStatus.FILLED, BrokerOrderStatus.CANCELLED, BrokerOrderStatus.REJECTED}
)
_RANK = {
    BrokerOrderStatus.SUBMITTED: 0,
    BrokerOrderStatus.UNKNOWN: 1,
    BrokerOrderStatus.PARTIALLY_FILLED: 2,
    BrokerOrderStatus.FILLED: 3,
    BrokerOrderStatus.CANCELLED: 3,
    BrokerOrderStatus.REJECTED: 3,
}


@dataclass(frozen=True, slots=True)
class BrokerCallback:
    event_id: str
    broker_order_id: str
    status: BrokerOrderStatus
    occurred_at: datetime

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.broker_order_id.strip():
            raise ValueError("broker callback ids must not be blank")
        object.__setattr__(
            self,
            "occurred_at",
            require_aware(self.occurred_at, field_name="occurred_at"),
        )


class BrokerEventJournal:
    """Archives every unique callback while preventing state regression from reordering."""

    def __init__(self) -> None:
        self._event_ids: set[str] = set()
        self._callbacks: list[BrokerCallback] = []
        self._statuses: dict[str, BrokerOrderStatus] = {}

    def record(self, callback: BrokerCallback) -> bool:
        if callback.event_id in self._event_ids:
            return False
        self._event_ids.add(callback.event_id)
        self._callbacks.append(callback)
        current = self._statuses.get(callback.broker_order_id)
        if current is None or (
            current not in _TERMINAL and _RANK[callback.status] >= _RANK[current]
        ):
            self._statuses[callback.broker_order_id] = callback.status
            return True
        return False

    def status(self, broker_order_id: str) -> BrokerOrderStatus | None:
        return self._statuses.get(broker_order_id)

    @property
    def callbacks(self) -> tuple[BrokerCallback, ...]:
        return tuple(self._callbacks)
