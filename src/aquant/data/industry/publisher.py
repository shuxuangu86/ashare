import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from aquant.data.history.tushare import TushareHistoryCatalog
from aquant.data.industry.models import IndustryClassification, IndustryMembershipRecord
from aquant.data.industry.standardizer import standardize_industry_history


@dataclass(frozen=True, slots=True)
class IndustryPublishResult:
    release_directory: Path
    manifest_path: Path
    content_hash: str
    status: str


class IndustryPITPublisher:
    SCHEMA_VERSION = "aquant.industry-pit-release.v1"

    def __init__(
        self,
        *,
        catalog: TushareHistoryCatalog,
        standard_root: Path,
        release_id: str,
        data_release_id: str,
        trading_days: tuple[date, ...],
        start_date: date,
        end_date: date,
        systems: tuple[str, ...] = ("SW2014", "SW2021"),
        code_version: str = "working-tree",
    ) -> None:
        if not release_id.strip() or "/" in release_id or "\\" in release_id:
            raise ValueError("industry release id must be a plain path component")
        if start_date > end_date:
            raise ValueError("industry release date range is invalid")
        if not trading_days or tuple(sorted(set(trading_days))) != trading_days:
            raise ValueError("industry release requires an ordered unique trading calendar")
        self._catalog = catalog
        self._root = standard_root
        self._release_id = release_id.strip()
        self._data_release_id = data_release_id.strip()
        self._trading_days = trading_days
        self._start = start_date
        self._end = end_date
        self._systems = tuple(system.upper() for system in systems)
        self._code_version = code_version

    def publish(self) -> IndustryPublishResult:
        final = self._root / f"industry-release={self._release_id}"
        if final.exists():
            return self._load_existing(final)
        building = self._root / f".industry-release={self._release_id}.building"
        building.mkdir(parents=True, exist_ok=True)
        standardized = standardize_industry_history(
            self._catalog,
            data_release_id=self._data_release_id,
            trading_days=self._trading_days,
            systems=self._systems,
            start_date=self._start,
            end_date=self._end,
        )
        classifications_path = building / "industry_classifications.parquet"
        memberships_path = building / "industry_memberships_pit.parquet"
        quarantine_path = building / "quarantine.parquet"
        _write_parquet_atomic(
            classifications_path,
            _classification_table(standardized.classifications),
        )
        _write_parquet_atomic(
            memberships_path,
            _membership_table(standardized.memberships),
        )
        _write_parquet_atomic(
            quarantine_path,
            _quarantine_table(standardized.quarantine),
        )
        reason_counts = Counter(str(item["reason_code"]) for item in standardized.quarantine)
        level_counts = Counter(
            (item.classification_version, item.industry_level) for item in standardized.memberships
        )
        hard_failures = {
            reason: count
            for reason, count in reason_counts.items()
            if reason
            in {
                "CONFLICTING_CLASSIFICATION",
                "CONFLICTING_MEMBERSHIP",
                "OVERLAPPING_MEMBERSHIP",
                "INVALID_MEMBERSHIP_DATE",
            }
            and count
        }
        missing_systems = [
            system
            for system in self._systems
            if not any(
                item.classification_version == system and item.industry_level == "L1"
                for item in standardized.memberships
            )
        ]
        status = "PASS" if not hard_failures and not missing_systems else "BLOCKED"
        files = {
            path.name: _file_hash(path)
            for path in (classifications_path, memberships_path, quarantine_path)
        }
        config = {
            "systems": self._systems,
            "start_date": self._start.isoformat(),
            "end_date": self._end.isoformat(),
            "availability_policy": "effective_session_close",
            "interval": "[effective_from,effective_to)",
        }
        config_hash = _hash(config)
        quality = {
            "status": status,
            "classification_count": len(standardized.classifications),
            "membership_count": len(standardized.memberships),
            "raw_classification_rows": standardized.raw_classification_rows,
            "raw_membership_rows": standardized.raw_membership_rows,
            "duplicate_membership_rows": standardized.duplicate_membership_rows,
            "quarantine_count": len(standardized.quarantine),
            "quarantine_reason_counts": dict(sorted(reason_counts.items())),
            "membership_level_counts": {
                f"{version}:{level}": count
                for (version, level), count in sorted(level_counts.items())
            },
            "hard_failures": hard_failures,
            "missing_systems": missing_systems,
        }
        _write_json_atomic(building / "quality.json", quality)
        stable_manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "release_id": self._release_id,
            "data_release_id": self._data_release_id,
            "status": status,
            "start_date": self._start.isoformat(),
            "end_date": self._end.isoformat(),
            "classification_systems": self._systems,
            "code_version": self._code_version,
            "config_hash": config_hash,
            "source_fingerprint": standardized.source_fingerprint,
            "files": files,
            "quality": quality,
            "lineage": {
                "raw_state_path": str(self._catalog.state_path),
                "source_apis": ("index_classify", "index_member"),
                "availability_policy": "effective_session_close",
                "announced_at": "UNAVAILABLE_FROM_SOURCE",
                "interval": "[effective_from,effective_to)",
            },
        }
        manifest = {
            **stable_manifest,
            "created_at": datetime.now(UTC).isoformat(timespec="microseconds"),
            "content_hash": _hash(stable_manifest),
        }
        _write_json_atomic(building / "manifest.json", manifest)
        if status != "PASS":
            raise ValueError(
                f"industry PIT quality gate blocked release: {json.dumps(quality, sort_keys=True)}"
            )
        building.rename(final)
        return IndustryPublishResult(
            final,
            final / "manifest.json",
            str(manifest["content_hash"]),
            status,
        )

    def _load_existing(self, directory: Path) -> IndustryPublishResult:
        manifest_path = directory / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        content_hash = payload.pop("content_hash")
        payload.pop("created_at")
        if content_hash != _hash(payload):
            raise ValueError("industry release manifest content hash mismatch")
        if payload["status"] != "PASS":
            raise ValueError("existing industry release is not usable")
        for name, expected in payload["files"].items():
            if _file_hash(directory / name) != expected:
                raise ValueError(f"industry release file hash mismatch: {name}")
        return IndustryPublishResult(directory, manifest_path, content_hash, "PASS")


def _classification_table(records: tuple[IndustryClassification, ...]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("classification_system", pa.string(), nullable=False),
            pa.field("classification_version", pa.string(), nullable=False),
            pa.field("industry_code", pa.string(), nullable=False),
            pa.field("native_industry_code", pa.string(), nullable=False),
            pa.field("industry_name", pa.string(), nullable=False),
            pa.field("industry_level", pa.string(), nullable=False),
            pa.field("parent_industry_code", pa.string(), nullable=True),
            pa.field("source", pa.string(), nullable=False),
            pa.field("source_record_id", pa.string(), nullable=False),
            pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("data_release_id", pa.string(), nullable=False),
            pa.field("content_hash", pa.string(), nullable=False),
        ]
    )
    return pa.Table.from_pylist([asdict(record) for record in records], schema=schema)


def _membership_table(records: tuple[IndustryMembershipRecord, ...]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("classification_system", pa.string(), nullable=False),
            pa.field("classification_version", pa.string(), nullable=False),
            pa.field("industry_code", pa.string(), nullable=False),
            pa.field("industry_name", pa.string(), nullable=False),
            pa.field("industry_level", pa.string(), nullable=False),
            pa.field("parent_industry_code", pa.string(), nullable=True),
            pa.field("instrument_id", pa.string(), nullable=False),
            pa.field("ts_code", pa.string(), nullable=False),
            pa.field("effective_from", pa.date32(), nullable=False),
            pa.field("effective_to", pa.date32(), nullable=True),
            pa.field("announced_at", pa.timestamp("us", tz="Asia/Shanghai"), nullable=True),
            pa.field("available_at", pa.timestamp("us", tz="Asia/Shanghai"), nullable=False),
            pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("source_record_id", pa.string(), nullable=False),
            pa.field("revision_no", pa.int32(), nullable=False),
            pa.field("data_release_id", pa.string(), nullable=False),
            pa.field("content_hash", pa.string(), nullable=False),
        ]
    )
    return pa.Table.from_pylist([asdict(record) for record in records], schema=schema)


def _quarantine_table(records: tuple[dict[str, object], ...]) -> pa.Table:
    return pa.Table.from_pydict(
        {
            "reason_code": [str(record["reason_code"]) for record in records],
            "record_json": [
                json.dumps(record["record"], ensure_ascii=False, sort_keys=True, default=str)
                for record in records
            ],
        },
        schema=pa.schema(
            [
                pa.field("reason_code", pa.string(), nullable=False),
                pa.field("record_json", pa.string(), nullable=False),
            ]
        ),
    )


def _write_parquet_atomic(path: Path, table: pa.Table) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_hash(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
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
