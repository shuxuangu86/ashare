import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from aquant.data.history.tushare import HistoryDatasetManifest, HistoryReleaseManifest
from aquant.strategies.all_a_equal_proxy import build_all_a_equal_weight_proxy
from aquant.strategies.nested_l4 import build_l4_eligibility_matrix


def test_all_a_proxy_is_pit_equal_weighted_and_carries_suspensions(tmp_path: Path) -> None:
    _release(tmp_path)

    points, metadata = build_all_a_equal_weight_proxy(
        tmp_path,
        start_date=date(2021, 1, 4),
        end_date=date(2021, 1, 6),
    )

    assert [point.constituent_count for point in points] == [1, 2, 2]
    assert [point.return_rate for point in points] == pytest.approx([0.10, 0.0, 0.0])
    assert metadata["official_wind_index"] is False
    assert metadata["minimum_constituents"] == 1
    assert metadata["maximum_constituents"] == 2
    eligibility = build_l4_eligibility_matrix(
        tmp_path,
        trade_dates=(date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)),
        ts_codes=("000001.SZ", "600001.SH"),
    )
    assert eligibility.tolist() == [[False, True], [False, False], [False, True]]


def _release(root: Path) -> None:
    datasets = {
        "daily": [
            {
                "trade_date": date(2020, 12, 31),
                "ts_code": "600001.SH",
                "exchange": "XSHG",
                "close": 100.0,
            },
            {
                "trade_date": date(2021, 1, 4),
                "ts_code": "600001.SH",
                "exchange": "XSHG",
                "close": 110.0,
            },
            {
                "trade_date": date(2021, 1, 5),
                "ts_code": "000001.SZ",
                "exchange": "XSHE",
                "close": 200.0,
            },
            {
                "trade_date": date(2021, 1, 6),
                "ts_code": "600001.SH",
                "exchange": "XSHG",
                "close": 121.0,
            },
            {
                "trade_date": date(2021, 1, 6),
                "ts_code": "000001.SZ",
                "exchange": "XSHE",
                "close": 180.0,
            },
        ],
        "adj_factor": [
            {"trade_date": date(2020, 12, 31), "ts_code": "600001.SH", "adj_factor": 1.0},
            {"trade_date": date(2021, 1, 4), "ts_code": "600001.SH", "adj_factor": 1.0},
            {"trade_date": date(2021, 1, 5), "ts_code": "000001.SZ", "adj_factor": 1.0},
            {"trade_date": date(2021, 1, 6), "ts_code": "600001.SH", "adj_factor": 1.0},
            {"trade_date": date(2021, 1, 6), "ts_code": "000001.SZ", "adj_factor": 1.0},
        ],
        "stock_basic": [
            {
                "ts_code": "600001.SH",
                "exchange": "XSHG",
                "list_date": date(2020, 1, 1),
                "delist_date": date(2022, 1, 1),
            },
            {
                "ts_code": "000001.SZ",
                "exchange": "XSHE",
                "list_date": date(2021, 1, 5),
                "delist_date": date(2022, 1, 1),
            },
        ],
        "namechange": [
            {
                "ts_code": "600001.SH",
                "name": "正常股份",
                "start_date": date(2020, 1, 1),
                "end_date": None,
            },
            {
                "ts_code": "000001.SZ",
                "name": "新股",
                "start_date": date(2021, 1, 5),
                "end_date": None,
            },
        ],
    }
    release_entries = []
    for name, rows in datasets.items():
        directory = root / f"dataset={name}"
        directory.mkdir(parents=True)
        path = directory / "data.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        manifest = HistoryDatasetManifest(
            schema_version="aquant.history-dataset.v1",
            dataset=name,
            row_count=len(rows),
            min_date="2020-12-31",
            max_date="2021-01-06",
            files=((path.name, hashlib.sha256(path.read_bytes()).hexdigest(), len(rows)),),
            source_page_count=1,
            source_fingerprint="1" * 64,
            transformations=(),
        )
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(manifest.to_json(), encoding="utf-8")
        release_entries.append(
            (name, hashlib.sha256(manifest_path.read_bytes()).hexdigest(), len(rows))
        )
    release = HistoryReleaseManifest(
        schema_version="aquant.history-release.v1",
        release_id="proxy-test",
        provider="TEST",
        start_date="2020-12-31",
        end_date="2021-01-06",
        created_at=datetime.now(UTC).isoformat(),
        datasets=tuple(sorted(release_entries)),
        raw_state_path="",
        status="PASS",
        caveats=(),
    )
    (root / "manifest.json").write_text(release.to_json(), encoding="utf-8")
