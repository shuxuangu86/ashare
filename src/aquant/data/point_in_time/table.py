from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Generic, TypeVar

from aquant.domain.time import require_aware

KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")


@dataclass(frozen=True, slots=True)
class PointInTimeRecord(Generic[KeyT, ValueT]):
    key: KeyT
    value: ValueT
    available_at: datetime
    ingested_at: datetime
    revision_no: int
    source: str
    source_record_id: str
    period_end: date | None = None
    announcement_date: datetime | None = None

    def __post_init__(self) -> None:
        available_at = require_aware(self.available_at, field_name="available_at")
        ingested_at = require_aware(self.ingested_at, field_name="ingested_at")
        announcement_date = self.announcement_date
        if announcement_date is not None:
            announcement_date = require_aware(announcement_date, field_name="announcement_date")
            if available_at < announcement_date:
                raise ValueError("available_at cannot be earlier than announcement_date")
        if ingested_at < available_at:
            raise ValueError("ingested_at cannot be earlier than available_at")
        if self.revision_no < 0:
            raise ValueError("revision_no cannot be negative")
        if not self.source.strip() or not self.source_record_id.strip():
            raise ValueError("source and source_record_id must not be blank")
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "ingested_at", ingested_at)
        object.__setattr__(self, "announcement_date", announcement_date)
        object.__setattr__(self, "source", self.source.strip().lower())
        object.__setattr__(self, "source_record_id", self.source_record_id.strip())


class PointInTimeTable(Generic[KeyT, ValueT]):
    def __init__(self, records: Iterable[PointInTimeRecord[KeyT, ValueT]]) -> None:
        resolved = tuple(records)
        identities: set[tuple[KeyT, str, str, int]] = set()
        for record in resolved:
            identity = (
                record.key,
                record.source,
                record.source_record_id,
                record.revision_no,
            )
            if identity in identities:
                raise ValueError(f"duplicate point-in-time record identity: {identity}")
            identities.add(identity)
        self._records = resolved

    def records_asof(self, asof_time: datetime) -> tuple[PointInTimeRecord[KeyT, ValueT], ...]:
        normalized_asof = require_aware(asof_time, field_name="asof_time")
        return tuple(record for record in self._records if record.available_at <= normalized_asof)

    def latest_by_key(self, asof_time: datetime) -> dict[KeyT, PointInTimeRecord[KeyT, ValueT]]:
        latest: dict[KeyT, PointInTimeRecord[KeyT, ValueT]] = {}
        for record in self.records_asof(asof_time):
            current = latest.get(record.key)
            if current is None or self._precedence(record) > self._precedence(current):
                latest[record.key] = record
        return latest

    def latest(self, key: KeyT, asof_time: datetime) -> PointInTimeRecord[KeyT, ValueT] | None:
        return self.latest_by_key(asof_time).get(key)

    @staticmethod
    def _precedence(record: PointInTimeRecord[KeyT, ValueT]) -> tuple[datetime, int, datetime, str]:
        return (
            record.available_at,
            record.revision_no,
            record.ingested_at,
            record.source_record_id,
        )
