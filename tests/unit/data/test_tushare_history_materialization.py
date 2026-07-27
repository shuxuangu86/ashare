import hashlib
import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from aquant.data.history import (
    DuckDBMicrocapHistory,
    TushareHistoryCatalog,
    TushareHistoryMaterializer,
)
from aquant.data.history.tushare import (
    _is_no_trade_daily_record,
    _price_limits,
    _stock_market,
)


def _state(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE pages (
            job_key TEXT PRIMARY KEY,
            api_name TEXT NOT NULL,
            params_json TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            row_count INTEGER,
            fields_json TEXT,
            payload_path TEXT,
            manifest_path TEXT,
            error TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()


def _page(
    state: Path,
    root: Path,
    *,
    api_name: str,
    params: dict[str, Any],
    fields: list[str],
    items: list[list[Any]],
    updated_at: datetime,
) -> None:
    batch_id = str(uuid4())
    directory = root / batch_id
    directory.mkdir(parents=True)
    body = json.dumps(
        {"code": 0, "data": {"fields": fields, "items": items}},
        separators=(",", ":"),
    ).encode()
    payload = directory / "response.json"
    payload.write_bytes(body)
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "aquant.raw-batch.v1",
                "batch_id": batch_id,
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    connection = sqlite3.connect(state)
    connection.execute(
        """
        INSERT INTO pages(
            job_key, api_name, params_json, status, attempts, row_count, fields_json,
            payload_path, manifest_path, error, updated_at
        ) VALUES (?, ?, ?, 'COMPLETED', 1, ?, ?, ?, ?, NULL, ?)
        """,
        (
            str(uuid4()),
            api_name,
            canonical,
            len(items),
            json.dumps(fields),
            str(payload),
            str(manifest),
            updated_at.isoformat(),
        ),
    )
    connection.commit()
    connection.close()


def test_catalog_selects_latest_complete_pagination_family(tmp_path: Path) -> None:
    state = tmp_path / "state.sqlite3"
    raw = tmp_path / "raw"
    _state(state)
    fields = ["ts_code", "trade_date"]
    base = {"trade_date": "20260717"}
    old = datetime(2026, 7, 20, tzinfo=UTC)
    new = datetime(2026, 7, 21, tzinfo=UTC)
    _page(
        state,
        raw,
        api_name="suspend_d",
        params={**base, "limit": 1, "offset": 0},
        fields=fields,
        items=[["000001.SZ", "20260717"]],
        updated_at=old,
    )
    _page(
        state,
        raw,
        api_name="suspend_d",
        params={**base, "limit": 1, "offset": 1},
        fields=fields,
        items=[],
        updated_at=old,
    )
    _page(
        state,
        raw,
        api_name="suspend_d",
        params={**base, "limit": 2, "offset": 0},
        fields=fields,
        items=[["000002.SZ", "20260717"]],
        updated_at=new,
    )

    pages = TushareHistoryCatalog(state).pages("suspend_d")

    assert len(pages) == 1
    assert pages[0].limit == 2
    assert pages[0].row_count == 1


def test_catalog_reconstructs_latest_family_from_duplicate_offsets(tmp_path: Path) -> None:
    state = tmp_path / "state.sqlite3"
    raw = tmp_path / "raw"
    _state(state)
    fields = ["exchange", "cal_date", "is_open"]
    base = {"start_date": "19900101", "end_date": "20260717"}
    old = datetime(2026, 7, 19, tzinfo=UTC)
    new = datetime(2026, 7, 20, tzinfo=UTC)
    for updated_at, first_day in ((old, "19900101"), (new, "19900102")):
        _page(
            state,
            raw,
            api_name="trade_cal",
            params={**base, "limit": 1, "offset": 0},
            fields=fields,
            items=[["SSE", first_day, "1"]],
            updated_at=updated_at,
        )
        _page(
            state,
            raw,
            api_name="trade_cal",
            params={**base, "limit": 1, "offset": 1},
            fields=fields,
            items=[],
            updated_at=updated_at + timedelta(seconds=1),
        )

    pages = TushareHistoryCatalog(state).pages("trade_cal")

    assert len(pages) == 1
    assert pages[0].updated_at == new


def test_catalog_rejects_duplicate_offsets_without_a_complete_family(tmp_path: Path) -> None:
    state = tmp_path / "state.sqlite3"
    raw = tmp_path / "raw"
    _state(state)
    fields = ["exchange", "cal_date", "is_open"]
    base = {"start_date": "19900101", "end_date": "20260717"}
    for index in range(2):
        _page(
            state,
            raw,
            api_name="trade_cal",
            params={**base, "limit": 1, "offset": 0},
            fields=fields,
            items=[["SSE", f"1990010{index + 1}", "1"]],
            updated_at=datetime(2026, 7, 19 + index, tzinfo=UTC),
        )

    try:
        TushareHistoryCatalog(state).pages("trade_cal")
    except ValueError as exc:
        assert "no complete pagination family" in str(exc)
    else:
        raise AssertionError("duplicate non-terminal pages must not be accepted as complete")


def test_catalog_accepts_fieldless_empty_terminal_after_full_page(tmp_path: Path) -> None:
    state = tmp_path / "state.sqlite3"
    raw = tmp_path / "raw"
    _state(state)
    now = datetime(2026, 7, 20, tzinfo=UTC)
    base = {"trade_date": "20211229"}
    fields = ["trade_date", "ts_code", "up_limit", "down_limit"]
    _page(
        state,
        raw,
        api_name="stk_limit",
        params={**base, "limit": 1, "offset": 0},
        fields=fields,
        items=[["20211229", "000001.SZ", 10, 9]],
        updated_at=now,
    )
    _page(
        state,
        raw,
        api_name="stk_limit",
        params={**base, "limit": 1, "offset": 1},
        fields=[],
        items=[],
        updated_at=now + timedelta(seconds=1),
    )

    pages = TushareHistoryCatalog(state).pages("stk_limit", include_empty=True)

    assert [(page.offset, page.row_count) for page in pages] == [(0, 1), (1, 0)]


def test_catalog_rejects_fieldless_terminal_without_a_typed_page(tmp_path: Path) -> None:
    state = tmp_path / "state.sqlite3"
    raw = tmp_path / "raw"
    _state(state)
    now = datetime(2026, 7, 20, tzinfo=UTC)
    _page(
        state,
        raw,
        api_name="stk_limit",
        params={"trade_date": "20211229", "limit": 2, "offset": 2},
        fields=[],
        items=[],
        updated_at=now,
    )
    try:
        TushareHistoryCatalog(state).pages("stk_limit")
    except ValueError:
        pass
    else:
        raise AssertionError("a fieldless terminal requires a compatible typed page")


def test_catalog_accepts_fieldless_empty_first_page(tmp_path: Path) -> None:
    state = tmp_path / "state.sqlite3"
    raw = tmp_path / "raw"
    _state(state)
    _page(
        state,
        raw,
        api_name="stk_limit",
        params={"trade_date": "19901219", "limit": 5800, "offset": 0},
        fields=[],
        items=[],
        updated_at=datetime(2026, 7, 20, tzinfo=UTC),
    )

    pages = TushareHistoryCatalog(state).pages("stk_limit", include_empty=True)

    assert len(pages) == 1
    assert pages[0].row_count == 0


def test_catalog_ignores_incomplete_global_calendar_when_both_exchanges_complete(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.sqlite3"
    raw = tmp_path / "raw"
    _state(state)
    fields = ["exchange", "cal_date", "is_open", "pretrade_date"]
    base = {"start_date": "19900101", "end_date": "20260717"}
    now = datetime(2026, 7, 20, tzinfo=UTC)
    _page(
        state,
        raw,
        api_name="trade_cal",
        params={**base, "limit": 1, "offset": 0},
        fields=fields,
        items=[["SSE", "19900101", "1", ""]],
        updated_at=now,
    )
    for index, exchange in enumerate(("SSE", "SZSE"), start=1):
        _page(
            state,
            raw,
            api_name="trade_cal",
            params={**base, "exchange": exchange, "limit": 1, "offset": 0},
            fields=fields,
            items=[[exchange, "19900101", "1", ""]],
            updated_at=now + timedelta(minutes=index),
        )
        _page(
            state,
            raw,
            api_name="trade_cal",
            params={**base, "exchange": exchange, "limit": 1, "offset": 1},
            fields=fields,
            items=[],
            updated_at=now + timedelta(minutes=index, seconds=1),
        )

    pages = TushareHistoryCatalog(state).pages("trade_cal")

    assert {page.base_params["exchange"] for page in pages} == {"SSE", "SZSE"}
    assert all("exchange" in page.base_params for page in pages)


def test_catalog_rejects_incomplete_global_calendar_without_both_exchanges(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.sqlite3"
    raw = tmp_path / "raw"
    _state(state)
    fields = ["exchange", "cal_date", "is_open", "pretrade_date"]
    base = {"start_date": "19900101", "end_date": "20260717"}
    now = datetime(2026, 7, 20, tzinfo=UTC)
    for params, items, updated_at in (
        ({**base, "limit": 1, "offset": 0}, [["SSE", "19900101", "1", ""]], now),
        (
            {**base, "exchange": "SSE", "limit": 1, "offset": 0},
            [["SSE", "19900101", "1", ""]],
            now + timedelta(minutes=1),
        ),
        (
            {**base, "exchange": "SSE", "limit": 1, "offset": 1},
            [],
            now + timedelta(minutes=1, seconds=1),
        ),
    ):
        _page(
            state,
            raw,
            api_name="trade_cal",
            params=params,
            fields=fields,
            items=items,
            updated_at=updated_at,
        )

    try:
        TushareHistoryCatalog(state).pages("trade_cal")
    except ValueError as exc:
        message = str(exc)
        assert "api_name=trade_cal" in message
        assert '"offset": 0' in message
        assert '"row_count": 1' in message
        assert '"fields"' in message
    else:
        raise AssertionError("a global calendar cannot replace a missing exchange family")


def test_catalog_rejects_global_calendar_when_exchange_fields_are_incomplete(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.sqlite3"
    raw = tmp_path / "raw"
    _state(state)
    base = {"start_date": "19900101", "end_date": "20260717"}
    now = datetime(2026, 7, 20, tzinfo=UTC)
    for index, exchange in enumerate(("", "SSE", "SZSE")):
        params = {
            **base,
            "limit": 1 if not exchange else 2,
            "offset": 0,
        }
        if exchange:
            params["exchange"] = exchange
        _page(
            state,
            raw,
            api_name="trade_cal",
            params=params,
            fields=["exchange", "cal_date"],
            items=[[exchange or "SSE", "19900101"]],
            updated_at=now + timedelta(minutes=index),
        )

    try:
        TushareHistoryCatalog(state).pages("trade_cal")
    except ValueError as exc:
        assert "api_name=trade_cal" in str(exc)
    else:
        raise AssertionError("exchange families missing is_open must fail closed")


def _daily_redundancy_fixture(
    state: Path,
    raw: Path,
    *,
    daily_dates: tuple[str, ...],
) -> None:
    now = datetime(2026, 7, 20, tzinfo=UTC)
    calendar_fields = ["exchange", "cal_date", "is_open", "pretrade_date"]
    for index, exchange in enumerate(("SSE", "SZSE")):
        _page(
            state,
            raw,
            api_name="trade_cal",
            params={
                "start_date": "20260716",
                "end_date": "20260717",
                "exchange": exchange,
                "limit": 3,
                "offset": 0,
            },
            fields=calendar_fields,
            items=[
                [exchange, "20260716", "1", "20260715"],
                [exchange, "20260717", "1", "20260716"],
            ],
            updated_at=now + timedelta(minutes=index),
        )
    daily_fields = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    ]
    _page(
        state,
        raw,
        api_name="daily",
        params={"start_date": "20260716", "end_date": "20260717", "limit": 1, "offset": 0},
        fields=daily_fields,
        items=[["000001.SZ", "20260716", 1, 1, 1, 1, 1, 1]],
        updated_at=now,
    )
    for index, trade_date in enumerate(daily_dates, start=1):
        _page(
            state,
            raw,
            api_name="daily",
            params={"trade_date": trade_date, "limit": 2, "offset": 0},
            fields=daily_fields,
            items=[["000001.SZ", trade_date, 1, 1, 1, 1, 1, 1]],
            updated_at=now + timedelta(minutes=index),
        )


def test_catalog_ignores_incomplete_global_daily_with_exact_calendar_coverage(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.sqlite3"
    raw = tmp_path / "raw"
    _state(state)
    _daily_redundancy_fixture(
        state,
        raw,
        daily_dates=("20260716", "20260717"),
    )

    pages = TushareHistoryCatalog(state).pages("daily")

    assert {page.base_params["trade_date"] for page in pages} == {
        "20260716",
        "20260717",
    }


def test_catalog_rejects_incomplete_global_daily_with_missing_open_date(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.sqlite3"
    raw = tmp_path / "raw"
    _state(state)
    _daily_redundancy_fixture(state, raw, daily_dates=("20260716",))

    try:
        TushareHistoryCatalog(state).pages("daily")
    except ValueError as exc:
        assert "api_name=daily" in str(exc)
    else:
        raise AssertionError("a missing open-market daily family must fail closed")


def test_no_trade_daily_placeholder_is_identified_strictly() -> None:
    placeholder = {
        "ts_code": "600717.SH",
        "trade_date": "20260610",
        "open": 0,
        "high": 0,
        "low": 0,
        "close": 4.31,
        "pre_close": 4.31,
        "vol": 0,
        "amount": 0,
        "change": 0,
        "pct_chg": 0,
    }

    assert _is_no_trade_daily_record(placeholder)

    traded = {**placeholder, "open": 4.31, "high": 4.31, "low": 4.31, "vol": 1}
    assert not _is_no_trade_daily_record(traded)


def test_malformed_zero_price_daily_record_fails_closed() -> None:
    malformed = {
        "ts_code": "600717.SH",
        "trade_date": "20260610",
        "open": 0,
        "high": 4.31,
        "low": 4.31,
        "close": 4.31,
        "pre_close": 4.31,
        "vol": 1,
        "amount": 1,
        "change": 0,
        "pct_chg": 0,
    }
    try:
        _is_no_trade_daily_record(malformed)
    except ValueError as exc:
        assert "600717.SH" in str(exc)
    else:
        raise AssertionError("mixed zero-price records must fail closed")


def test_zero_price_limits_represent_an_unlimited_session() -> None:
    assert _price_limits(0, 0) == (None, None)
    assert _price_limits(99999.99, 0) == (None, None)
    assert _price_limits(99999.99, None) == (None, None)
    assert _price_limits(100000, None) == (None, None)
    assert _price_limits(100000, 0) == (None, None)
    assert _price_limits(11, 9) == (Decimal("11.000000"), Decimal("9.000000"))


def test_one_sided_zero_price_limit_fails_closed() -> None:
    try:
        _price_limits(0, 9)
    except ValueError as exc:
        assert "up_limit=0" in str(exc)
    else:
        raise AssertionError("one-sided zero price limits must fail closed")


def test_missing_legacy_stock_market_is_explicitly_unknown() -> None:
    assert _stock_market(None) == "UNKNOWN"
    assert _stock_market("Main Board") == "Main Board"


def test_suspend_types_collapse_to_one_symbol_date_presence_row(tmp_path: Path) -> None:
    state = tmp_path / "state.sqlite3"
    raw = tmp_path / "raw"
    standard = tmp_path / "standard"
    _state(state)
    _page(
        state,
        raw,
        api_name="suspend_d",
        params={"trade_date": "20110523", "limit": 10, "offset": 0},
        fields=["ts_code", "trade_date", "suspend_type", "suspend_timing"],
        items=[
            ["600572.SH", "20110523", "R", None],
            ["600572.SH", "20110523", "S", None],
        ],
        updated_at=datetime(2026, 7, 20, tzinfo=UTC),
    )
    result = TushareHistoryMaterializer(
        catalog=TushareHistoryCatalog(state),
        standard_root=standard,
        release_id="test",
        start_date=date(2011, 5, 23),
        end_date=date(2011, 5, 23),
        verify_raw_checksums=True,
    ).materialize(("suspend_d",))

    table = pq.read_table(
        result.release_directory / "dataset=suspend_d" / "year=2011" / "data.parquet",
    )

    assert table.num_rows == 1
    table = table.select(["ts_code", "trade_date"])
    assert table.to_pylist() == [
        {
            "ts_code": "600572.SH",
            "trade_date": date(2011, 5, 23),
        }
    ]


def test_materializer_corrects_units_windows_and_stock_metadata(tmp_path: Path) -> None:
    state = tmp_path / "state.sqlite3"
    raw = tmp_path / "raw"
    _state(state)
    started = date(2026, 1, 1)
    now = datetime(2026, 7, 20, tzinfo=UTC)
    daily_fields = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    ]
    basic_fields = [
        "ts_code",
        "trade_date",
        "total_mv",
        "circ_mv",
        "pb",
        "turnover_rate",
        "dv_ttm",
    ]
    for index in range(21):
        session = (started + timedelta(days=index)).strftime("%Y%m%d")
        _page(
            state,
            raw,
            api_name="daily",
            params={"trade_date": session, "limit": 6000, "offset": 0},
            fields=daily_fields,
            items=[
                [
                    "000001.SZ",
                    session,
                    10,
                    11,
                    9,
                    10.5,
                    index + 1,
                    index + 2,
                ]
            ],
            updated_at=now,
        )
        _page(
            state,
            raw,
            api_name="daily_basic",
            params={"trade_date": session, "limit": 6000, "offset": 0},
            fields=basic_fields,
            items=[["000001.SZ", session, 12.5, 10, 1.2, index + 1, 2.5]],
            updated_at=now,
        )
    stock_fields = ["ts_code", "symbol", "name", "market", "list_date"]
    _page(
        state,
        raw,
        api_name="stock_basic",
        params={"list_status": "D", "limit": 6000, "offset": 0},
        fields=stock_fields,
        items=[["000001.SZ", "000001", "平安银行", "主板", "19910403"]],
        updated_at=now,
    )

    result = TushareHistoryMaterializer(
        catalog=TushareHistoryCatalog(state),
        standard_root=tmp_path / "standard",
        release_id="test-history",
        start_date=started + timedelta(days=20),
        end_date=started + timedelta(days=20),
    ).materialize(("daily", "daily_basic", "stock_basic"))

    daily_path = (
        result.release_directory
        / "dataset=daily"
        / f"year={(started + timedelta(days=20)).year}"
        / "data.parquet"
    )
    daily = pq.ParquetFile(daily_path).read().to_pylist()
    assert len(daily) == 1
    assert str(daily[0]["volume"]) == "2100.0000"
    assert str(daily[0]["amount"]) == "22000.0000"
    assert str(daily[0]["prior_20d_average_volume"]) == "1050.0000"

    basic_path = (
        result.release_directory
        / "dataset=daily_basic"
        / f"year={(started + timedelta(days=20)).year}"
        / "data.parquet"
    )
    basic = pq.ParquetFile(basic_path).read().to_pylist()
    assert str(basic[0]["total_market_cap"]) == "125000.0000"
    assert basic[0]["turnover_volatility_20d"] is not None

    instruments = (
        pq.ParquetFile(result.release_directory / "dataset=stock_basic" / "data.parquet")
        .read()
        .to_pylist()
    )
    assert instruments == [
        {
            "ts_code": "000001.SZ",
            "symbol": "000001",
            "name": "平安银行",
            "market": "主板",
            "exchange": "XSHE",
            "list_status": "D",
            "list_date": date(1991, 4, 3),
            "delist_date": started + timedelta(days=20),
            "delist_date_inferred": True,
        }
    ]
    assert result.manifest.status == "MATERIALIZED_NOT_BACKTEST_APPROVED"


def test_materializer_is_idempotent_and_rejects_window_reuse(tmp_path: Path) -> None:
    state = tmp_path / "state.sqlite3"
    raw = tmp_path / "raw"
    _state(state)
    now = datetime(2026, 7, 20, tzinfo=UTC)
    fields = ["ts_code", "trade_date", "up_limit", "down_limit"]
    _page(
        state,
        raw,
        api_name="stk_limit",
        params={"trade_date": "20260717", "limit": 5800, "offset": 0},
        fields=fields,
        items=[["000001.SZ", "20260717", 11, 9]],
        updated_at=now,
    )
    catalog = TushareHistoryCatalog(state)
    first = TushareHistoryMaterializer(
        catalog=catalog,
        standard_root=tmp_path / "standard",
        release_id="same",
        start_date=date(2026, 7, 17),
        end_date=date(2026, 7, 17),
    ).materialize(("stk_limit",))
    second = TushareHistoryMaterializer(
        catalog=catalog,
        standard_root=tmp_path / "standard",
        release_id="same",
        start_date=date(2026, 7, 17),
        end_date=date(2026, 7, 17),
    ).materialize(("stk_limit",))

    assert first.manifest == second.manifest

    mismatched = TushareHistoryMaterializer(
        catalog=catalog,
        standard_root=tmp_path / "standard",
        release_id="same",
        start_date=date(2026, 7, 16),
        end_date=date(2026, 7, 17),
    )
    try:
        mismatched.materialize(("stk_limit",))
    except ValueError as exc:
        assert "date window" in str(exc)
    else:
        raise AssertionError("release id reuse with a different window must fail")


def test_history_reader_enforces_fundamental_lag_and_builds_sessions(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.sqlite3"
    raw = tmp_path / "raw"
    _state(state)
    now = datetime(2026, 7, 20, tzinfo=UTC)
    trade_date = date(2026, 7, 17)
    value = trade_date.strftime("%Y%m%d")
    _page(
        state,
        raw,
        api_name="daily",
        params={"trade_date": value, "limit": 6000, "offset": 0},
        fields=[
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
        ],
        items=[["000001.SZ", value, 10, 11, 9, 10.5, 100, 1000]],
        updated_at=now,
    )
    _page(
        state,
        raw,
        api_name="daily_basic",
        params={"trade_date": value, "limit": 6000, "offset": 0},
        fields=[
            "ts_code",
            "trade_date",
            "total_mv",
            "circ_mv",
            "pb",
            "turnover_rate",
            "dv_ttm",
        ],
        items=[["000001.SZ", value, 100, 80, 1.5, 2, 3]],
        updated_at=now,
    )
    _page(
        state,
        raw,
        api_name="stock_basic",
        params={"list_status": "L", "limit": 6000, "offset": 0},
        fields=["ts_code", "symbol", "name", "market", "list_date"],
        items=[["000001.SZ", "000001", "平安银行", "主板", "19910403"]],
        updated_at=now,
    )
    _page(
        state,
        raw,
        api_name="namechange",
        params={"limit": 5000, "offset": 0},
        fields=["ts_code", "name", "start_date", "end_date", "ann_date"],
        items=[["000001.SZ", "平安银行", "19910403", None, "19910403"]],
        updated_at=now,
    )
    _page(
        state,
        raw,
        api_name="fina_indicator",
        params={"ts_code": "000001.SZ", "limit": 5000, "offset": 0},
        fields=[
            "ts_code",
            "ann_date",
            "end_date",
            "netprofit_yoy",
            "debt_to_assets",
            "update_flag",
        ],
        items=[
            ["000001.SZ", "20260716", "20260331", 12, 40, "1"],
            ["000001.SZ", "20260717", "20260630", 99, 99, "1"],
        ],
        updated_at=now,
    )
    _page(
        state,
        raw,
        api_name="suspend_d",
        params={"trade_date": value, "limit": 5000, "offset": 0},
        fields=["ts_code", "trade_date"],
        items=[["000001.SZ", value]],
        updated_at=now,
    )
    _page(
        state,
        raw,
        api_name="stk_limit",
        params={"trade_date": value, "limit": 5800, "offset": 0},
        fields=["ts_code", "trade_date", "up_limit", "down_limit"],
        items=[["000001.SZ", value, 11, 9]],
        updated_at=now,
    )
    _page(
        state,
        raw,
        api_name="trade_cal",
        params={
            "exchange": "SZSE",
            "start_date": value,
            "end_date": value,
            "limit": 6000,
            "offset": 0,
        },
        fields=["exchange", "cal_date", "is_open"],
        items=[["SZSE", value, "1"]],
        updated_at=now,
    )
    datasets = (
        "daily",
        "daily_basic",
        "stk_limit",
        "suspend_d",
        "stock_basic",
        "namechange",
        "trade_cal",
        "fina_indicator",
    )
    release = TushareHistoryMaterializer(
        catalog=TushareHistoryCatalog(state),
        standard_root=tmp_path / "standard",
        release_id="reader",
        start_date=trade_date,
        end_date=trade_date,
    ).materialize(datasets)

    with DuckDBMicrocapHistory(release.release_directory) as history:
        assert history.trading_dates(trade_date, trade_date) == (trade_date,)
        snapshot = history.snapshot(trade_date)
        sessions = history.sessions(trade_date, trade_date)

    observation = snapshot.observations[0]
    assert observation.net_profit_yoy == 12
    assert observation.debt_to_assets == 40
    assert observation.suspended
    assert not observation.is_st
    assert sessions[0].statuses[0].suspended
    assert sessions[0].bars[0].volume == 10_000
