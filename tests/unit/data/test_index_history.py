import hashlib
import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from aquant.data.history.tushare import RawHistoryPage, TushareHistoryCatalog
from aquant.data.index_history import (
    IndexHistory,
    IndexWeightSnapshot,
    _decimal,
    _optional_date,
    _rows,
    load_index_history,
)
from aquant.domain.enums import Exchange
from aquant.domain.identifiers import Symbol


def _page(tmp_path: Path, rows: list[list[object]]) -> RawHistoryPage:
    fields = ["index_code", "con_code", "trade_date", "weight"]
    body = json.dumps({"code": 0, "data": {"fields": fields, "items": rows}}).encode()
    payload = tmp_path / "response.json"
    manifest = tmp_path / "manifest.json"
    payload.write_bytes(body)
    manifest.write_text(
        json.dumps({"sha256": hashlib.sha256(body).hexdigest()}),
        encoding="utf-8",
    )
    return RawHistoryPage(
        "index_weight",
        {
            "index_code": "000300.SH",
            "start_date": "20240101",
            "end_date": "20240131",
        },
        len(rows),
        tuple(fields),
        payload,
        manifest,
        datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_raw_index_rows_verify_hash_and_filter_dates(tmp_path: Path) -> None:
    page = _page(
        tmp_path,
        [
            ["000300.SH", "600000.SH", "20240102", 60],
            ["000300.SH", "000001.SZ", "20240102", 40],
        ],
    )
    rows, hashes = _rows(
        (page,),
        index_code="000300.SH",
        start_date=datetime(2024, 1, 1).date(),
        end_date=datetime(2024, 1, 31).date(),
    )
    assert len(rows) == 2
    assert len(hashes) == 1
    page.payload_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        _rows(
            (page,),
            index_code="000300.SH",
            start_date=datetime(2024, 1, 1).date(),
            end_date=datetime(2024, 1, 31).date(),
        )


def test_index_weight_snapshot_enforces_unique_positive_hundred_percent() -> None:
    symbol = Symbol("600000", Exchange.XSHG)
    with pytest.raises(ValueError, match="sum"):
        IndexWeightSnapshot(datetime(2024, 1, 2).date(), ((symbol, 90),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate"):
        IndexWeightSnapshot(
            datetime(2024, 1, 2).date(),
            ((symbol, 50), (symbol, 50)),  # type: ignore[arg-type]
        )


class _Catalog:
    def __init__(self, pages: dict[str, tuple[RawHistoryPage, ...]]) -> None:
        self._pages = pages

    def pages(self, api_name: str) -> tuple[RawHistoryPage, ...]:
        return self._pages.get(api_name, ())


def _daily_page(tmp_path: Path, rows: list[list[object]]) -> RawHistoryPage:
    fields = ["ts_code", "trade_date", "close"]
    body = json.dumps({"code": 0, "data": {"fields": fields, "items": rows}}).encode()
    payload = tmp_path / "daily-response.json"
    manifest = tmp_path / "daily-manifest.json"
    payload.write_bytes(body)
    manifest.write_text(
        json.dumps({"sha256": hashlib.sha256(body).hexdigest()}),
        encoding="utf-8",
    )
    return RawHistoryPage(
        "index_daily",
        {
            "ts_code": "000300.SH",
            "start_date": "20240101",
            "end_date": "20240131",
        },
        len(rows),
        tuple(fields),
        payload,
        manifest,
        datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_load_index_history_builds_verified_ordered_history(tmp_path: Path) -> None:
    history = load_index_history(
        cast(
            TushareHistoryCatalog,
            _Catalog(
                {
                    "index_weight": (
                        _page(
                            tmp_path,
                            [
                                ["000300.SH", "600000.SH", "20240102", 60],
                                ["000300.SH", "000001.SZ", "20240102", 40],
                            ],
                        ),
                    ),
                    "index_daily": (
                        _daily_page(
                            tmp_path,
                            [
                                ["000300.SH", "20240102", 3300],
                                ["000300.SH", "20240103", 3333],
                            ],
                        ),
                    ),
                }
            ),
        ),
        index_code=" 000300.sh ",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )
    assert history.index_code == "000300.SH"
    assert history.closes[-1] == (date(2024, 1, 3), Decimal("3333"))
    assert len(history.weight_snapshots[0].weights) == 2
    assert len(history.source_fingerprint) == 64


def test_index_history_rejects_conflicts_ranges_and_invalid_values(tmp_path: Path) -> None:
    duplicate_daily = _daily_page(
        tmp_path,
        [
            ["000300.SH", "20240102", 3300],
            ["000300.SH", "20240102", 3301],
        ],
    )
    catalog = cast(
        TushareHistoryCatalog,
        _Catalog(
            {
                "index_weight": (
                    _page(
                        tmp_path,
                        [
                            ["000300.SH", "600000.SH", "20240102", 60],
                            ["000300.SH", "000001.SZ", "20240102", 40],
                        ],
                    ),
                ),
                "index_daily": (duplicate_daily,),
            }
        ),
    )
    with pytest.raises(ValueError, match="conflicting index close"):
        load_index_history(
            catalog,
            index_code="000300.SH",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )
    with pytest.raises(ValueError, match="date range"):
        load_index_history(
            catalog,
            index_code="000300.SH",
            start_date=date(2024, 2, 1),
            end_date=date(2024, 1, 31),
        )
    for value in ("NaN", "0", "-1"):
        with pytest.raises(ValueError, match="finite and positive"):
            _decimal(value)
    assert _optional_date(None) is None
    assert _optional_date("") is None


def test_index_history_requires_complete_ordered_inputs() -> None:
    symbol = Symbol("600000", Exchange.XSHG)
    snapshot = IndexWeightSnapshot(
        date(2024, 1, 2),
        ((symbol, Decimal("100")),),
    )
    with pytest.raises(ValueError, match="incomplete"):
        IndexHistory("", (snapshot,), ((date(2024, 1, 2), Decimal("1")),), "hash")
    with pytest.raises(ValueError, match="snapshots must be ordered"):
        IndexHistory(
            "000300.SH",
            (
                IndexWeightSnapshot(date(2024, 1, 3), snapshot.weights),
                snapshot,
            ),
            ((date(2024, 1, 2), Decimal("1")),),
            "hash",
        )
    with pytest.raises(ValueError, match="closes must be ordered"):
        IndexHistory(
            "000300.SH",
            (snapshot,),
            (
                (date(2024, 1, 3), Decimal("2")),
                (date(2024, 1, 2), Decimal("1")),
            ),
            "hash",
        )
