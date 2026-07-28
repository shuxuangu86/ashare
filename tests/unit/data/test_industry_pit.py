import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from hypothesis import given
from hypothesis import strategies as st

from aquant.data.industry.models import IndustryClassification, IndustryMembershipRecord
from aquant.data.industry.publisher import _classification_table, _hash, _membership_table
from aquant.data.industry.quality import (
    IndustryQualityThresholds,
    validate_industry_release,
)
from aquant.data.industry.repository import IndustryPITRepository
from aquant.data.industry.standardizer import _remove_overlaps

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _membership(
    *,
    industry: str = "801010.SI",
    effective_from: date = date(2021, 12, 13),
    effective_to: date | None = date(2022, 1, 5),
    available_at: datetime = datetime(2021, 12, 13, 15, tzinfo=SHANGHAI),
) -> IndustryMembershipRecord:
    return IndustryMembershipRecord(
        classification_system="SW",
        classification_version="SW2021",
        industry_code=industry,
        industry_name="农林牧渔",
        industry_level="L1",
        parent_industry_code=None,
        instrument_id="000001.SZ",
        ts_code="000001.SZ",
        effective_from=effective_from,
        effective_to=effective_to,
        announced_at=None,
        available_at=available_at,
        ingested_at=datetime(2026, 7, 17, tzinfo=UTC),
        source="tushare",
        source_record_id=f"{industry}:{effective_from}",
        revision_no=0,
        data_release_id="test-release",
        content_hash=f"{industry}:{effective_from}:{effective_to}",
    )


def _repository(
    tmp_path: Path,
    records: tuple[IndustryMembershipRecord, ...],
) -> IndustryPITRepository:
    release = tmp_path / "industry-release=test"
    release.mkdir()
    membership = release / "industry_memberships_pit.parquet"
    pq.write_table(_membership_table(records), membership)
    file_hash = hashlib.sha256(membership.read_bytes()).hexdigest()
    stable = {
        "schema_version": "aquant.industry-pit-release.v1",
        "release_id": "test",
        "data_release_id": "test-release",
        "status": "PASS",
        "files": {membership.name: file_hash},
    }
    manifest = {
        **stable,
        "created_at": "2026-07-17T00:00:00+00:00",
        "content_hash": _hash(stable),
    }
    (release / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return IndustryPITRepository(release)


def test_membership_interval_is_left_closed_right_open() -> None:
    record = _membership()
    assert record.contains(date(2021, 12, 13))
    assert record.contains(date(2022, 1, 4))
    assert not record.contains(date(2022, 1, 5))


def test_pit_query_requires_available_at(tmp_path: Path) -> None:
    repository = _repository(tmp_path, (_membership(),))
    assert (
        repository.latest(
            "000001.SZ",
            date(2021, 12, 13),
            datetime(2021, 12, 13, 14, 59, tzinfo=SHANGHAI),
        )
        is None
    )
    assert (
        repository.latest(
            "000001.SZ",
            date(2021, 12, 13),
            datetime(2021, 12, 13, 15, tzinfo=SHANGHAI),
        )
        is not None
    )


def test_future_revision_does_not_change_past_panel(tmp_path: Path) -> None:
    original = _membership()
    future = _membership(
        industry="801020.SI",
        effective_from=date(2022, 1, 5),
        effective_to=None,
        available_at=datetime(2022, 1, 5, 15, tzinfo=SHANGHAI),
    )
    repository = _repository(tmp_path, (original, future))
    panel = repository.panel(
        (date(2022, 1, 4), date(2022, 1, 5)),
        ("000001.SZ",),
    )
    assert panel.industries[:, 0].tolist() == ["801010.SI", "801020.SI"]


def test_panel_rejects_duplicate_dates(tmp_path: Path) -> None:
    repository = _repository(tmp_path, (_membership(),))
    with pytest.raises(ValueError, match="ordered and unique"):
        repository.panel(
            (date(2022, 1, 4), date(2022, 1, 4)),
            ("000001.SZ",),
        )


def test_overlap_is_quarantined() -> None:
    left = _membership(effective_to=date(2022, 1, 10))
    right = _membership(
        industry="801020.SI",
        effective_from=date(2022, 1, 5),
        effective_to=None,
        available_at=datetime(2022, 1, 5, 15, tzinfo=SHANGHAI),
    )
    accepted, quarantine = _remove_overlaps((left, right))
    assert accepted == ()
    assert quarantine[0]["reason_code"] == "OVERLAPPING_MEMBERSHIP"


@given(st.dates(min_value=date(2000, 1, 2), max_value=date(2030, 12, 30)))
def test_effective_to_is_never_visible(end: date) -> None:
    record = replace(
        _membership(),
        effective_from=end.fromordinal(end.toordinal() - 1),
        effective_to=end,
    )
    assert record.contains(end.fromordinal(end.toordinal() - 1))
    assert not record.contains(end)


def test_repository_rejects_naive_as_of(tmp_path: Path) -> None:
    repository = _repository(tmp_path, (_membership(),))
    with pytest.raises(ValueError, match="timezone-aware"):
        repository.latest("000001.SZ", date(2022, 1, 4), datetime(2022, 1, 4, 15))


def test_panel_can_use_configured_evaluation_time(tmp_path: Path) -> None:
    repository = _repository(tmp_path, (_membership(),))
    panel = repository.panel(
        (date(2021, 12, 13),),
        ("000001.SZ",),
        evaluation_time=time(14, 59),
    )
    assert panel.industries[0, 0] is None


def test_quality_gate_detects_annual_coverage_gap(tmp_path: Path) -> None:
    repository = _repository(tmp_path, (_membership(effective_to=None),))
    release = repository.release_directory
    classification = IndustryClassification(
        classification_system="SW",
        classification_version="SW2021",
        industry_code="801010.SI",
        native_industry_code="110000",
        industry_name="农林牧渔",
        industry_level="L1",
        parent_industry_code=None,
        source="tushare",
        source_record_id="classification",
        ingested_at=datetime(2026, 7, 17, tzinfo=UTC),
        data_release_id="test-release",
        content_hash="classification",
    )
    pq.write_table(
        _classification_table((classification,)),
        release / "industry_classifications.parquet",
    )
    history = tmp_path / "history" / "dataset=daily" / "year=2022"
    history.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "ts_code": "000001.SZ",
                    "exchange": "XSHE",
                    "trade_date": date(2022, 1, 4),
                },
                {
                    "ts_code": "000002.SZ",
                    "exchange": "XSHE",
                    "trade_date": date(2022, 1, 4),
                },
                {
                    "ts_code": "920001.BJ",
                    "exchange": "XBSE",
                    "trade_date": date(2022, 1, 4),
                },
            ]
        ),
        history / "data.parquet",
    )
    report = validate_industry_release(
        industry_release=release,
        history_release=tmp_path / "history",
        start_date=date(2022, 1, 1),
        end_date=date(2022, 12, 31),
        thresholds=IndustryQualityThresholds(
            minimum_market_row_coverage=0.4,
            minimum_year_market_row_coverage=0.75,
        ),
    )
    assert report["market_rows"] == 2
    assert report["status"] == "BLOCKED"
    assert report["failure_reason_codes"] == ["YEAR_MARKET_ROW_COVERAGE_BELOW_THRESHOLD"]
