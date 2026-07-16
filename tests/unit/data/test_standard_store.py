import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pyarrow.parquet as pq
import pytest

from aquant.data.normalization.standard_store import (
    DailyBarParquetStore,
    StandardBatchManifest,
)
from aquant.domain.enums import Exchange
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar

BATCH_ID = UUID("22222222-2222-2222-2222-222222222222")
RAW_BATCH_ID = UUID("11111111-1111-1111-1111-111111111111")
STANDARDIZED_AT = datetime(2026, 7, 16, 9, tzinfo=UTC)


def _bar(code: str, exchange: Exchange, trade_date: date, close: str) -> DailyBar:
    close_value = Decimal(close)
    return DailyBar(
        symbol=Symbol(code, exchange),
        trade_date=trade_date,
        open=close_value - Decimal("0.10"),
        high=close_value + Decimal("0.20"),
        low=close_value - Decimal("0.20"),
        close=close_value,
        volume=Decimal("100000"),
        amount=Decimal("1020000"),
    )


def _bars() -> tuple[DailyBar, ...]:
    return (
        _bar("600000", Exchange.XSHG, date(2026, 7, 16), "10.20"),
        _bar("000001", Exchange.XSHE, date(2026, 7, 15), "12.30"),
    )


def test_standard_store_writes_schema_enforced_parquet_with_lineage(tmp_path: Path) -> None:
    result = DailyBarParquetStore(tmp_path).write(
        _bars(),
        provider="akshare",
        raw_batch_ids=(RAW_BATCH_ID,),
        standardized_at=STANDARDIZED_AT,
        batch_id=BATCH_ID,
    )

    assert result.directory == (
        tmp_path / "dataset=bars_1d" / "provider=akshare" / f"batch_id={BATCH_ID}"
    )
    assert result.manifest.schema_version == "aquant.standard.bars-1d.v1"
    assert result.manifest.raw_batch_ids == (str(RAW_BATCH_ID),)
    assert result.manifest.row_count == 2
    assert result.manifest.min_trade_date == "2026-07-15"
    assert result.manifest.max_trade_date == "2026-07-16"
    assert len(result.manifest.sha256) == 64

    table = pq.ParquetFile(result.data_path).read()
    assert table.column_names == [
        "symbol",
        "exchange",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert table.schema.metadata is not None
    assert table.schema.metadata[b"raw_batch_ids"] == str(RAW_BATCH_ID).encode()


def test_standard_store_round_trip_is_lossless_and_sorted(tmp_path: Path) -> None:
    store = DailyBarParquetStore(tmp_path)
    result = store.write(
        _bars(),
        provider="akshare",
        raw_batch_ids=(RAW_BATCH_ID,),
        standardized_at=STANDARDIZED_AT,
        batch_id=BATCH_ID,
    )

    restored = store.read(result.directory)

    assert restored == tuple(sorted(_bars(), key=lambda item: (item.trade_date, item.symbol)))


def test_standard_store_detects_data_corruption(tmp_path: Path) -> None:
    store = DailyBarParquetStore(tmp_path)
    result = store.write(
        _bars(),
        provider="akshare",
        raw_batch_ids=(RAW_BATCH_ID,),
        standardized_at=STANDARDIZED_AT,
        batch_id=BATCH_ID,
    )
    with result.data_path.open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(ValueError, match="checksum mismatch"):
        store.read(result.directory)


def test_standard_store_never_overwrites_batch(tmp_path: Path) -> None:
    store = DailyBarParquetStore(tmp_path)
    kwargs = {
        "provider": "akshare",
        "raw_batch_ids": (RAW_BATCH_ID,),
        "standardized_at": STANDARDIZED_AT,
        "batch_id": BATCH_ID,
    }
    store.write(_bars(), **kwargs)  # type: ignore[arg-type]
    with pytest.raises(FileExistsError, match="already exists"):
        store.write(_bars(), **kwargs)  # type: ignore[arg-type]


def test_standard_store_rejects_missing_lineage_empty_and_duplicate_rows(tmp_path: Path) -> None:
    store = DailyBarParquetStore(tmp_path)
    with pytest.raises(ValueError, match="must not be empty"):
        store.write(
            (),
            provider="akshare",
            raw_batch_ids=(RAW_BATCH_ID,),
            standardized_at=STANDARDIZED_AT,
        )
    with pytest.raises(ValueError, match="at least one raw batch"):
        store.write(
            _bars(),
            provider="akshare",
            raw_batch_ids=(),
            standardized_at=STANDARDIZED_AT,
        )
    with pytest.raises(ValueError, match="duplicate primary keys"):
        store.write(
            (_bars()[0], _bars()[0]),
            provider="akshare",
            raw_batch_ids=(RAW_BATCH_ID,),
            standardized_at=STANDARDIZED_AT,
        )


@pytest.mark.parametrize("provider", ["../escape", "AK/SHARE", ""])
def test_standard_store_rejects_unsafe_provider_partition(tmp_path: Path, provider: str) -> None:
    with pytest.raises(ValueError, match="unsafe provider"):
        DailyBarParquetStore(tmp_path).write(
            _bars(),
            provider=provider,
            raw_batch_ids=(RAW_BATCH_ID,),
            standardized_at=STANDARDIZED_AT,
        )


def test_standard_manifest_requires_raw_batch_list() -> None:
    payload = {
        "schema_version": "aquant.standard.bars-1d.v1",
        "batch_id": str(BATCH_ID),
        "dataset": "bars_1d",
        "provider": "akshare",
        "raw_batch_ids": "not-a-list",
        "standardized_at": STANDARDIZED_AT.isoformat(),
        "data_file": "data.parquet",
        "sha256": "0" * 64,
        "row_count": 0,
        "min_trade_date": "2026-07-16",
        "max_trade_date": "2026-07-16",
    }
    with pytest.raises(ValueError, match="must be a list"):
        StandardBatchManifest.from_json(json.dumps(payload))
