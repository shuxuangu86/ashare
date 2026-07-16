import json
from datetime import date
from pathlib import Path

import pytest

from aquant.data.normalization import (
    TusharePayloadError,
    normalize_instruments,
    normalize_trading_calendar,
    parse_tabular_payload,
)
from aquant.domain.enums import Board, Exchange

FIXTURES = Path(__file__).parents[2] / "fixtures" / "providers" / "demo"


def test_normalize_stock_basic_to_canonical_instruments() -> None:
    instruments = normalize_instruments((FIXTURES / "tushare_stock_basic.json").read_bytes())

    assert [item.symbol.canonical for item in instruments] == [
        "600000.XSHG",
        "688001.XSHG",
        "300001.XSHE",
    ]
    assert [item.board for item in instruments] == [Board.MAIN, Board.STAR, Board.CHINEXT]
    assert instruments[0].list_date == date(1999, 11, 10)
    assert instruments[0].delist_date is None


def test_normalize_trade_calendar_to_exchange_sessions() -> None:
    sessions = normalize_trading_calendar((FIXTURES / "tushare_trade_cal.json").read_bytes())

    assert sessions[0].exchange is Exchange.XSHG
    assert sessions[0].trade_date == date(2026, 7, 15)
    assert sessions[0].is_open is True
    assert sessions[2].is_open is False
    assert sessions[3].exchange is Exchange.XSHE


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not-json", "not valid JSON"),
        (b"[]", "root must be an object"),
        (json.dumps({"code": 2002, "msg": "denied"}).encode(), "API error"),
        (json.dumps({"code": 0, "data": []}).encode(), "data must be an object"),
        (
            json.dumps({"code": 0, "data": {"fields": "symbol", "items": []}}).encode(),
            "fields must be a list",
        ),
        (
            json.dumps({"code": 0, "data": {"fields": [1], "items": []}}).encode(),
            "fields must be a list",
        ),
        (
            json.dumps({"code": 0, "data": {"fields": [], "items": {}}}).encode(),
            "items must be a list",
        ),
        (
            json.dumps({"code": 0, "data": {"fields": ["symbol"], "items": [[]]}}).encode(),
            "does not match fields",
        ),
    ],
)
def test_parse_tabular_payload_rejects_malformed_data(payload: bytes, message: str) -> None:
    with pytest.raises(TusharePayloadError, match=message):
        parse_tabular_payload(payload)


def _payload(fields: list[str], item: list[object]) -> bytes:
    return json.dumps({"code": 0, "data": {"fields": fields, "items": [item]}}).encode()


def test_instrument_normalizer_rejects_unsupported_exchange() -> None:
    fields = ["symbol", "name", "market", "exchange", "list_date", "delist_date"]
    with pytest.raises(TusharePayloadError, match="unsupported exchange"):
        normalize_instruments(
            _payload(fields, ["920001", "北交测试", "北交所", "BSE", "20260101", None])
        )


@pytest.mark.parametrize(
    "item",
    [
        ["600000", "", "主板", "SSE", "19991110", None],
        ["600000", "浦发银行", "主板", "SSE", "1999-11-10", None],
        ["600000", "浦发银行", "主板", "SSE", "19991340", None],
        ["600000", "浦发银行", "主板", "SSE", "19991110", "bad-date"],
    ],
)
def test_instrument_normalizer_rejects_invalid_required_values(item: list[object]) -> None:
    fields = ["symbol", "name", "market", "exchange", "list_date", "delist_date"]
    with pytest.raises(TusharePayloadError):
        normalize_instruments(_payload(fields, item))


def test_unknown_market_maps_to_other_board() -> None:
    fields = ["symbol", "name", "market", "exchange", "list_date", "delist_date"]
    instrument = normalize_instruments(
        _payload(fields, ["600000", "浦发银行", "CDR", "SSE", "19991110", None])
    )[0]
    assert instrument.board is Board.OTHER


def test_calendar_normalizer_rejects_invalid_open_flag() -> None:
    with pytest.raises(TusharePayloadError, match="is_open"):
        normalize_trading_calendar(
            _payload(
                ["exchange", "cal_date", "is_open", "pretrade_date"],
                ["SSE", "20260716", 2, "20260715"],
            )
        )
