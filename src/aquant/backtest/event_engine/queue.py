import heapq
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generic, TypeVar

from aquant.domain.time import require_aware

EventT = TypeVar("EventT")


@dataclass(order=True, frozen=True, slots=True)
class ScheduledEvent(Generic[EventT]):
    occurred_at: datetime
    priority: int
    sequence: int
    event: EventT = field(compare=False)


class EventQueue(Generic[EventT]):
    """Stable priority queue ordered by time, priority, then insertion sequence."""

    def __init__(self) -> None:
        self._heap: list[ScheduledEvent[EventT]] = []
        self._next_sequence = 0
        self._last_popped_at: datetime | None = None

    def push(
        self, event: EventT, *, occurred_at: datetime, priority: int
    ) -> ScheduledEvent[EventT]:
        timestamp = require_aware(occurred_at, field_name="occurred_at")
        if self._last_popped_at is not None and timestamp < self._last_popped_at:
            raise ValueError("cannot schedule an event before the current backtest clock")
        scheduled = ScheduledEvent(timestamp, priority, self._next_sequence, event)
        self._next_sequence += 1
        heapq.heappush(self._heap, scheduled)
        return scheduled

    def pop(self) -> ScheduledEvent[EventT]:
        if not self._heap:
            raise IndexError("event queue is empty")
        scheduled = heapq.heappop(self._heap)
        if self._last_popped_at is not None and scheduled.occurred_at < self._last_popped_at:
            raise RuntimeError("event queue clock moved backwards")
        self._last_popped_at = scheduled.occurred_at
        return scheduled

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def current_time(self) -> datetime | None:
        return self._last_popped_at
