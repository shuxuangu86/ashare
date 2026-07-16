import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from aquant.domain.enums import Exchange
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar
from aquant.domain.time import require_aware

_PARTITION_VALUE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


@dataclass(frozen=True, slots=True)
class StandardBatchManifest:
    schema_version: str
    batch_id: str
    dataset: str
    provider: str
    raw_batch_ids: tuple[str, ...]
    standardized_at: str
    data_file: str
    sha256: str
    row_count: int
    min_trade_date: str
    max_trade_date: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "StandardBatchManifest":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("standard manifest root must be an object")
        raw_batch_ids = payload.get("raw_batch_ids")
        if not isinstance(raw_batch_ids, list):
            raise ValueError("raw_batch_ids must be a list")
        payload["raw_batch_ids"] = tuple(raw_batch_ids)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class StandardBatchResult:
    directory: Path
    data_path: Path
    manifest_path: Path
    manifest: StandardBatchManifest


class DailyBarParquetStore:
    """Append-only, schema-enforced Parquet storage for standardized daily bars."""

    DATASET = "bars_1d"
    SCHEMA_VERSION = "aquant.standard.bars-1d.v1"

    def __init__(self, root: Path) -> None:
        self._root = root

    def write(
        self,
        bars: tuple[DailyBar, ...],
        *,
        provider: str,
        raw_batch_ids: tuple[UUID, ...],
        standardized_at: datetime,
        batch_id: UUID | None = None,
    ) -> StandardBatchResult:
        if not bars:
            raise ValueError("standard daily-bar batch must not be empty")
        if not raw_batch_ids:
            raise ValueError("standard batch must reference at least one raw batch")
        provider = self._partition_value(provider)
        standardized_at = require_aware(standardized_at, field_name="standardized_at")
        keys = [(bar.symbol, bar.trade_date) for bar in bars]
        if len(keys) != len(set(keys)):
            raise ValueError("standard daily-bar batch contains duplicate primary keys")

        resolved_batch_id = batch_id or uuid4()
        parent = self._root / f"dataset={self.DATASET}" / f"provider={provider}"
        final_directory = parent / f"batch_id={resolved_batch_id}"
        if final_directory.exists():
            raise FileExistsError(f"standard batch already exists: {final_directory}")
        parent.mkdir(parents=True, exist_ok=True)
        temporary_directory = parent / f".{resolved_batch_id}.tmp-{uuid4().hex}"
        temporary_directory.mkdir()
        temporary_data_path = temporary_directory / "data.parquet"

        sorted_bars = tuple(sorted(bars, key=lambda item: (item.trade_date, item.symbol)))
        raw_ids = tuple(str(value) for value in sorted(set(raw_batch_ids), key=str))
        try:
            table = self._table(sorted_bars, raw_ids, standardized_at, resolved_batch_id, provider)
            pq.write_table(
                table,
                temporary_data_path,
                compression="zstd",
                use_dictionary=["symbol", "exchange"],
                write_statistics=True,
            )
            self._fsync(temporary_data_path)
            checksum = self._checksum(temporary_data_path)
            manifest = StandardBatchManifest(
                schema_version=self.SCHEMA_VERSION,
                batch_id=str(resolved_batch_id),
                dataset=self.DATASET,
                provider=provider,
                raw_batch_ids=raw_ids,
                standardized_at=standardized_at.isoformat(timespec="microseconds"),
                data_file="data.parquet",
                sha256=checksum,
                row_count=len(sorted_bars),
                min_trade_date=sorted_bars[0].trade_date.isoformat(),
                max_trade_date=sorted_bars[-1].trade_date.isoformat(),
            )
            manifest_path = temporary_directory / "manifest.json"
            self._write_new_file(manifest_path, manifest.to_json().encode("utf-8"))
            temporary_directory.rename(final_directory)
        except Exception:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            raise

        return StandardBatchResult(
            directory=final_directory,
            data_path=final_directory / "data.parquet",
            manifest_path=final_directory / "manifest.json",
            manifest=manifest,
        )

    def read(self, directory: Path, *, verify_checksum: bool = True) -> tuple[DailyBar, ...]:
        manifest = StandardBatchManifest.from_json(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        data_path = directory / manifest.data_file
        if verify_checksum and self._checksum(data_path) != manifest.sha256:
            raise ValueError("standard Parquet checksum mismatch")
        table = pq.ParquetFile(data_path).read()
        if table.schema.metadata is None or table.schema.metadata.get(b"schema_version") != (
            self.SCHEMA_VERSION.encode()
        ):
            raise ValueError("unexpected standard Parquet schema version")
        bars = tuple(
            DailyBar(
                symbol=Symbol(row["symbol"], Exchange(row["exchange"])),
                trade_date=row["trade_date"],
                open=Decimal(row["open"]),
                high=Decimal(row["high"]),
                low=Decimal(row["low"]),
                close=Decimal(row["close"]),
                volume=Decimal(row["volume"]),
                amount=Decimal(row["amount"]),
            )
            for row in table.to_pylist()
        )
        if len(bars) != manifest.row_count:
            raise ValueError("standard Parquet row count does not match manifest")
        return bars

    @classmethod
    def _table(
        cls,
        bars: tuple[DailyBar, ...],
        raw_batch_ids: tuple[str, ...],
        standardized_at: datetime,
        batch_id: UUID,
        provider: str,
    ) -> pa.Table:
        schema = pa.schema(
            [
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("exchange", pa.string(), nullable=False),
                pa.field("trade_date", pa.date32(), nullable=False),
                pa.field("open", pa.decimal128(20, 8), nullable=False),
                pa.field("high", pa.decimal128(20, 8), nullable=False),
                pa.field("low", pa.decimal128(20, 8), nullable=False),
                pa.field("close", pa.decimal128(20, 8), nullable=False),
                pa.field("volume", pa.decimal128(28, 4), nullable=False),
                pa.field("amount", pa.decimal128(28, 4), nullable=False),
            ],
            metadata={
                b"schema_version": cls.SCHEMA_VERSION.encode(),
                b"batch_id": str(batch_id).encode(),
                b"provider": provider.encode(),
                b"raw_batch_ids": ",".join(raw_batch_ids).encode(),
                b"standardized_at": standardized_at.isoformat().encode(),
            },
        )
        rows = [
            {
                "symbol": bar.symbol.code,
                "exchange": bar.symbol.exchange.value,
                "trade_date": bar.trade_date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "amount": bar.amount,
            }
            for bar in bars
        ]
        return pa.Table.from_pylist(rows, schema=schema)

    @staticmethod
    def _partition_value(value: str) -> str:
        normalized = value.strip().lower()
        if not _PARTITION_VALUE.fullmatch(normalized):
            raise ValueError(f"unsafe provider partition value: {value!r}")
        return normalized

    @staticmethod
    def _checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _fsync(path: Path) -> None:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())

    @staticmethod
    def _write_new_file(path: Path, body: bytes) -> None:
        with path.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
