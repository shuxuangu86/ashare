import hashlib
import json
from bisect import bisect_left
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from aquant.data.industry.models import IndustryMembershipRecord, IndustryPITPanel

_SHANGHAI = ZoneInfo("Asia/Shanghai")


class IndustryPITRepository:
    def __init__(self, release_directory: Path) -> None:
        self.release_directory = release_directory
        manifest_path = release_directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.pop("content_hash")
        manifest.pop("created_at")
        if expected != _hash(manifest) or manifest.get("status") != "PASS":
            raise ValueError("industry PIT release manifest is invalid")
        membership_path = release_directory / "industry_memberships_pit.parquet"
        if _file_hash(membership_path) != manifest["files"][membership_path.name]:
            raise ValueError("industry PIT membership content hash mismatch")
        self.data_release_id = str(manifest["data_release_id"])
        self.content_hash = str(expected)
        self._membership_path = membership_path

    def records(
        self,
        *,
        level: str = "L1",
        ts_codes: tuple[str, ...] | None = None,
    ) -> tuple[IndustryMembershipRecord, ...]:
        # ParquetFile avoids PyArrow interpreting the parent
        # ``industry-release=<id>`` directory as an extra Hive column.
        table = pq.ParquetFile(self._membership_path).read()
        requested = set(ts_codes or ())
        requested_level = level.upper()
        output: list[IndustryMembershipRecord] = []
        for row in table.to_pylist():
            if row["industry_level"] != requested_level:
                continue
            if requested and row["ts_code"] not in requested:
                continue
            output.append(IndustryMembershipRecord(**row))
        return tuple(output)

    def latest(
        self,
        ts_code: str,
        trade_date: date,
        as_of_time: datetime,
        *,
        level: str = "L1",
    ) -> IndustryMembershipRecord | None:
        if as_of_time.tzinfo is None or as_of_time.utcoffset() is None:
            raise ValueError("industry as_of_time must be timezone-aware")
        visible = [
            record
            for record in self.records(level=level, ts_codes=(ts_code,))
            if record.contains(trade_date) and record.available_at <= as_of_time
        ]
        if len(visible) > 1:
            raise ValueError("industry PIT query returned conflicting memberships")
        return visible[0] if visible else None

    def panel(
        self,
        trade_dates: tuple[date, ...],
        ts_codes: tuple[str, ...],
        *,
        level: str = "L1",
        evaluation_time: time = time(15),
    ) -> IndustryPITPanel:
        if tuple(sorted(set(trade_dates))) != trade_dates:
            raise ValueError("industry panel dates must be ordered and unique")
        if len(set(ts_codes)) != len(ts_codes):
            raise ValueError("industry panel securities must be unique")
        shape = (len(trade_dates), len(ts_codes))
        industries = np.full(shape, None, dtype=object)
        available_dates = np.full(shape, None, dtype=object)
        columns = {code: index for index, code in enumerate(ts_codes)}
        for record in self.records(level=level, ts_codes=ts_codes):
            column = columns[record.ts_code]
            start = bisect_left(trade_dates, record.effective_from)
            end = (
                len(trade_dates)
                if record.effective_to is None
                else bisect_left(trade_dates, record.effective_to)
            )
            for row in range(start, end):
                as_of = datetime.combine(trade_dates[row], evaluation_time, _SHANGHAI)
                if record.available_at > as_of:
                    continue
                current = industries[row, column]
                if current is not None and current != record.industry_code:
                    raise ValueError("industry panel contains overlapping memberships")
                industries[row, column] = record.industry_code
                available_dates[row, column] = record.available_at.date()
        return IndustryPITPanel(trade_dates, ts_codes, industries, available_dates)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()
