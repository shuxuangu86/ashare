import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from aquant.data.ingestion import RawBatchStore
from aquant.data.normalization import DailyBarParquetStore, normalize_daily_bars
from aquant.data.providers import AkshareProvider, DatasetRequest


class Frame:
    def __len__(self) -> int:
        return 1

    def to_json(self, **_: object) -> str:
        return json.dumps(
            {
                "data": [
                    {
                        "日期": "2026-07-16",
                        "股票代码": "600000",
                        "开盘": 10.1,
                        "最高": 10.5,
                        "最低": 10,
                        "收盘": 10.3,
                        "成交量": 1234,
                        "成交额": 1269000.25,
                    }
                ]
            },
            ensure_ascii=False,
        )


class Backend:
    def stock_zh_a_hist(self, **_: object) -> Frame:
        return Frame()


def test_akshare_raw_to_standard_pipeline(tmp_path: Path) -> None:
    requested_at = datetime(2026, 7, 16, 8, tzinfo=UTC)
    request = DatasetRequest.create(
        dataset="daily_bars",
        params={"symbol": "600000.XSHG", "start_date": "20260716", "end_date": "20260716"},
        requested_at=requested_at,
    )
    response = AkshareProvider(
        backend=Backend(),
        clock=lambda: requested_at + timedelta(seconds=1),
    ).fetch(request)
    raw = RawBatchStore(tmp_path / "raw").write_response(
        response,
        batch_id=UUID("00000000-0000-0000-0000-000000000001"),
    )
    bars = normalize_daily_bars(raw.payload_path.read_bytes())
    standard = DailyBarParquetStore(tmp_path / "standardized").write(
        bars,
        provider="akshare",
        raw_batch_ids=(raw.manifest.batch_id,),
        standardized_at=requested_at + timedelta(seconds=2),
        batch_id=UUID("00000000-0000-0000-0000-000000000002"),
    )

    restored = DailyBarParquetStore(tmp_path / "standardized").read(standard.directory)
    assert restored == bars
    assert standard.manifest.raw_batch_ids == (raw.manifest.batch_id,)
