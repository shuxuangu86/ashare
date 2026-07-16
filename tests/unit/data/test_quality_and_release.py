import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from aquant.data.catalog import DataReleasePublisher
from aquant.data.quality import DailyBarQualityGate, QualityIssue, QualityReport, Severity
from aquant.domain.data_release import DataReleaseId
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar, SecurityStatus

SYMBOL = Symbol.parse("600000.XSHG")
TRADE_DATE = date(2026, 7, 16)


def _bar(
    *,
    symbol: Symbol = SYMBOL,
    close: str = "10.3",
    volume: str = "123400",
    trade_date: date = TRADE_DATE,
) -> DailyBar:
    return DailyBar(
        symbol,
        trade_date,
        Decimal("10.1"),
        Decimal("10.5"),
        Decimal("10.0"),
        Decimal(close),
        Decimal(volume),
        Decimal("1269000"),
    )


def test_market_gate_detects_duplicates_calendar_and_suspension() -> None:
    status = SecurityStatus(SYMBOL, TRADE_DATE, suspended=True, is_st=False)
    report = DailyBarQualityGate().validate(
        (_bar(), _bar()),
        open_dates=(date(2026, 7, 15),),
        statuses=(status,),
    )
    assert not report.passed
    assert {issue.code for issue in report.issues} == {
        "DUPLICATE_PRIMARY_KEY",
        "NON_TRADING_DATE",
        "BAR_ON_SUSPENSION",
    }


def test_cross_source_gate_passes_tolerable_rounding() -> None:
    report = DailyBarQualityGate().compare_sources(
        (_bar(),),
        (_bar(close="10.301", volume="124000"),),
    )
    assert report.passed


def test_cross_source_gate_blocks_material_differences_and_missing_rows() -> None:
    other = Symbol.parse("000001.XSHE")
    report = DailyBarQualityGate().compare_sources(
        (_bar(), _bar(symbol=other)),
        (_bar(close="10.5", volume="100000"),),
    )
    assert {issue.code for issue in report.issues} == {
        "CLOSE_MISMATCH",
        "VOLUME_MISMATCH",
        "CROSS_SOURCE_MISSING",
    }


def test_cross_source_gate_rejects_bad_tolerance_and_duplicate_keys() -> None:
    gate = DailyBarQualityGate()
    with pytest.raises(ValueError, match="non-negative"):
        gate.compare_sources((), (), close_relative_tolerance=Decimal("-1"))
    with pytest.raises(ValueError, match="duplicate"):
        gate.compare_sources((_bar(), _bar()), (_bar(),))


def test_publishes_immutable_release_bound_to_files_and_reports(tmp_path: Path) -> None:
    data_file = tmp_path / "bars.parquet"
    data_file.write_bytes(b"immutable-standard-data")
    report = QualityReport("daily_bars", 1)
    publisher = DataReleasePublisher(tmp_path / "releases")
    release_id = DataReleaseId("cn_equity_20260716_001")

    release = publisher.publish(
        release_id,
        data_files=(data_file,),
        reports=(report,),
        created_at=datetime(2026, 7, 16, 20, tzinfo=UTC),
    )
    manifest = json.loads((release / "manifest.json").read_text())

    assert manifest["status"] == "PUBLISHED"
    assert manifest["release_id"] == str(release_id)
    assert manifest["data_files"][0][0] == "bars.parquet"
    assert len(manifest["data_files"][0][1]) == 64
    with pytest.raises(FileExistsError):
        publisher.publish(release_id, data_files=(data_file,), reports=(report,))


def test_failed_quality_report_is_quarantined_and_not_published(tmp_path: Path) -> None:
    data_file = tmp_path / "bars.parquet"
    data_file.write_bytes(b"bad-data")
    issue = QualityIssue(Severity.ERROR, "CLOSE_MISMATCH", "600000@2026-07-16", "bad")
    report = QualityReport("daily_bars", 1, (issue,))
    publisher = DataReleasePublisher(tmp_path / "releases")
    release_id = DataReleaseId("cn_equity_20260716_002")

    with pytest.raises(ValueError, match="blocked"):
        publisher.publish(release_id, data_files=(data_file,), reports=(report,))

    assert not (tmp_path / "releases" / str(release_id)).exists()
    quarantine = list((tmp_path / "quarantine").glob("*.json"))
    assert len(quarantine) == 1
    assert json.loads(quarantine[0].read_text())["release_id"] == str(release_id)


def test_release_requires_inputs_and_report_validates_fields(tmp_path: Path) -> None:
    publisher = DataReleasePublisher(tmp_path / "releases")
    release_id = DataReleaseId("cn_equity_20260716_003")
    with pytest.raises(ValueError, match="requires"):
        publisher.publish(release_id, data_files=(), reports=())
    with pytest.raises(ValueError):
        QualityReport("", -1)
    with pytest.raises(ValueError):
        QualityIssue(Severity.WARNING, "", "x", "message")
