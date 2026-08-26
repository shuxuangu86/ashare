import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from aquant.factors.materialization import (
    MaterializationManifest,
    MaterializedFactorReader,
)
from aquant.factors.materialization.manifest import MaterializedFile


def _release(tmp_path: Path) -> Path:
    root = tmp_path / "materialization=test"
    relative = (
        "factor_family=liquidity/year=2026/month=01/"
        "factor=amount_concentration_20d-version=1.0.0.parquet"
    )
    path = root / relative
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "trade_date": [date(2026, 1, 5), date(2026, 1, 5), date(2026, 1, 6)],
                "ts_code": ["000001.SZ", "600000.SH", "920001.BJ"],
                "factor_id": ["amount_concentration_20d"] * 3,
                "factor_version": ["1.0.0"] * 3,
                "value": [0.1, 0.2, 0.3],
                "is_valid": [True, True, True],
                "quality_flags": [[], [], []],
                "data_release_id": ["release"] * 3,
                "computed_at": [datetime(2026, 1, 6, tzinfo=UTC)] * 3,
            }
        ),
        path,
    )
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    MaterializationManifest(
        materialization_id="test",
        data_release_id="release",
        factor_versions=(("amount_concentration_20d", "1.0.0"),),
        start_date="2026-01-01",
        end_date="2026-01-31",
        universe="all_a_share",
        code_version="test",
        config_hash="a" * 64,
        computed_at="2026-01-31T15:00:00+08:00",
        files=(MaterializedFile(relative, 3, file_hash),),
        total_rows=3,
        content_hash=file_hash,
    ).write(root / "manifest.json")
    return root


def test_reader_verifies_partitions_and_filters_market_scope(tmp_path: Path) -> None:
    reader = MaterializedFactorReader(_release(tmp_path))
    values = reader.load(
        factor_id="amount_concentration_20d",
        factor_version="1.0.0",
        start_date=date(2026, 1, 5),
        end_date=date(2026, 1, 6),
    )
    assert [(item.ts_code, item.value) for item in values] == [
        ("000001.SZ", 0.1),
        ("600000.SH", 0.2),
    ]


def test_reader_rejects_tampered_partition(tmp_path: Path) -> None:
    root = _release(tmp_path)
    path = next(root.rglob("*.parquet"))
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="content hash"):
        MaterializedFactorReader(root).load(
            factor_id="amount_concentration_20d",
            factor_version="1.0.0",
            start_date=date(2026, 1, 5),
            end_date=date(2026, 1, 6),
        )
