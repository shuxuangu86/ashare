from datetime import UTC, date, datetime, timedelta

import pytest

from aquant.backtest.event_engine import (
    EventPriority,
    EventQueue,
    SessionCloseEvent,
    SessionOpenEvent,
)


def test_queue_orders_by_time_priority_and_insertion_sequence() -> None:
    queue: EventQueue[str] = EventQueue()
    start = datetime(2026, 7, 17, 1, 30, tzinfo=UTC)
    queue.push(
        "close", occurred_at=start + timedelta(hours=6), priority=EventPriority.SESSION_CLOSE
    )
    queue.push("order-1", occurred_at=start, priority=EventPriority.ORDER)
    queue.push("open", occurred_at=start, priority=EventPriority.SESSION_OPEN)
    queue.push("order-2", occurred_at=start, priority=EventPriority.ORDER)

    assert [queue.pop().event for _ in range(4)] == ["open", "order-1", "order-2", "close"]
    assert not queue
    assert queue.current_time == start + timedelta(hours=6)


def test_queue_rejects_naive_and_past_events() -> None:
    queue: EventQueue[str] = EventQueue()
    now = datetime(2026, 7, 17, tzinfo=UTC)
    with pytest.raises(ValueError, match="timezone"):
        queue.push("bad", occurred_at=datetime(2026, 7, 17), priority=1)
    queue.push("now", occurred_at=now, priority=1)
    queue.pop()
    with pytest.raises(ValueError, match="before"):
        queue.push("past", occurred_at=now - timedelta(seconds=1), priority=1)
    with pytest.raises(IndexError, match="empty"):
        queue.pop()


def test_session_events_require_aware_timestamps() -> None:
    opened = SessionOpenEvent(date(2026, 7, 17), datetime(2026, 7, 17, 1, 30, tzinfo=UTC))
    closed = SessionCloseEvent(date(2026, 7, 17), datetime(2026, 7, 17, 7, tzinfo=UTC))
    assert opened.kind.value == "SESSION_OPEN"
    assert closed.kind.value == "SESSION_CLOSE"
    with pytest.raises(ValueError, match="timezone"):
        SessionOpenEvent(date(2026, 7, 17), datetime(2026, 7, 17, 1, 30))
