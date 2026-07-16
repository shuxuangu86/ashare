from datetime import UTC, date, datetime

import pytest

from aquant.data.point_in_time import PointInTimeContext, PointInTimeRecord, PointInTimeTable
from aquant.domain.data_release import DataReleaseId


def _time(day: int) -> datetime:
    return datetime(2026, 7, day, 8, tzinfo=UTC)


def _record(
    *,
    key: str = "600000.XSHG:2026Q1",
    value: float = 1.0,
    available_day: int = 1,
    ingested_day: int = 2,
    revision_no: int = 0,
    source_record_id: str = "record-1",
) -> PointInTimeRecord[str, float]:
    return PointInTimeRecord(
        key=key,
        value=value,
        period_end=date(2026, 3, 31),
        announcement_date=_time(available_day),
        available_at=_time(available_day),
        ingested_at=_time(ingested_day),
        revision_no=revision_no,
        source="fixture",
        source_record_id=source_record_id,
    )


def test_point_in_time_query_never_returns_future_revision() -> None:
    original = _record(value=1.0, available_day=1, ingested_day=2)
    revision = _record(
        value=2.0,
        available_day=10,
        ingested_day=11,
        revision_no=1,
        source_record_id="record-2",
    )
    table = PointInTimeTable([original, revision])

    assert table.latest(original.key, _time(9)) == original
    assert table.latest(original.key, _time(10)) == revision
    assert table.latest(original.key, _time(1)).value == 1.0  # type: ignore[union-attr]


def test_point_in_time_snapshot_keeps_latest_record_per_key() -> None:
    first = _record()
    second = _record(key="000001.XSHE:2026Q1", source_record_id="record-2")
    table = PointInTimeTable([first, second])

    snapshot = table.latest_by_key(_time(9))

    assert snapshot == {first.key: first, second.key: second}
    assert table.latest("missing", _time(9)) is None


def test_point_in_time_context_binds_release_and_asof() -> None:
    record = _record()
    context = PointInTimeContext(
        asof_time=_time(9),
        data_release_id=DataReleaseId("cn_equity_20260716_001"),
        tables={"financials": PointInTimeTable([record])},
    )

    assert context.latest("financials", record.key) == record
    assert context.snapshot("financials") == {record.key: record}
    with pytest.raises(KeyError, match="not registered"):
        context.latest("arbitrary_table", record.key)
    with pytest.raises(KeyError, match="not registered"):
        context.snapshot("arbitrary_table")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"available_at": _time(1), "announcement_date": _time(2)},
        {"available_at": _time(2), "ingested_at": _time(1)},
        {"revision_no": -1},
        {"source": ""},
        {"source_record_id": ""},
    ],
)
def test_point_in_time_record_rejects_invalid_metadata(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "key": "key",
        "value": 1.0,
        "available_at": _time(2),
        "ingested_at": _time(3),
        "revision_no": 0,
        "source": "fixture",
        "source_record_id": "record-1",
        "announcement_date": _time(1),
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        PointInTimeRecord(**values)  # type: ignore[arg-type]


def test_point_in_time_table_rejects_duplicate_record_identity() -> None:
    record = _record()
    with pytest.raises(ValueError, match="duplicate"):
        PointInTimeTable([record, record])


def test_point_in_time_query_requires_aware_timestamp() -> None:
    table = PointInTimeTable([_record()])
    with pytest.raises(ValueError, match="timezone"):
        table.records_asof(datetime(2026, 7, 9))
