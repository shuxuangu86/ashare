from datetime import UTC, datetime

import pytest

from aquant.domain.audit import AuditEvent
from aquant.domain.data_release import DataReleaseId


def test_audit_payload_hash_is_canonical() -> None:
    first = AuditEvent.create(event_type="order.created", payload={"b": 2, "a": 1})
    second = AuditEvent.create(event_type="order.created", payload={"a": 1, "b": 2})

    assert first.payload_json == '{"a":1,"b":2}'
    assert first.payload_hash == second.payload_hash


def test_audit_event_normalizes_time_to_utc() -> None:
    event = AuditEvent.create(
        event_type="data.published",
        payload={"release": "cn_equity_20260716_001"},
        occurred_at=datetime(2026, 7, 16, 16, tzinfo=UTC),
    )

    assert event.occurred_at.tzinfo is UTC


def test_audit_event_rejects_blank_type() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        AuditEvent.create(event_type=" ", payload={})


@pytest.mark.parametrize(
    "value",
    [
        "cn_equity_20260716_001",
        "cn_equity_19900101_999",
    ],
)
def test_data_release_id_accepts_canonical_values(value: str) -> None:
    assert str(DataReleaseId(value)) == value


@pytest.mark.parametrize(
    "value", ["CN_EQUITY_20260716_001", "cn_equity_2026716_001", "cn_equity_20260716_1"]
)
def test_data_release_id_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="must match"):
        DataReleaseId(value)
