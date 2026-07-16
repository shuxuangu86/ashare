import json
from datetime import date
from decimal import Decimal

import pytest

from aquant.data.normalization import AksharePayloadError, normalize_daily_bars


def _payload(**overrides: object) -> bytes:
    record: dict[str, object] = {
        "日期": "2026-07-16T00:00:00.000",
        "股票代码": "600000",
        "开盘": 10.1,
        "收盘": 10.3,
        "最高": 10.5,
        "最低": 10.0,
        "成交量": 1234,
        "成交额": 1269000.25,
    }
    record.update(overrides)
    return json.dumps({"schema": {}, "data": [record]}, ensure_ascii=False).encode()


def test_normalizes_akshare_units_and_identifiers() -> None:
    (bar,) = normalize_daily_bars(_payload())

    assert bar.symbol.canonical == "600000.XSHG"
    assert bar.trade_date == date(2026, 7, 16)
    assert bar.open == Decimal("10.1")
    assert bar.close == Decimal("10.3")
    assert bar.volume == Decimal("123400")
    assert bar.amount == Decimal("1269000.25")


@pytest.mark.parametrize(
    ("code", "canonical"),
    [
        ("688001", "688001.XSHG"),
        ("000001", "000001.XSHE"),
        ("300001", "300001.XSHE"),
    ],
)
def test_maps_supported_exchange_prefixes(code: str, canonical: str) -> None:
    (bar,) = normalize_daily_bars(_payload(**{"股票代码": code}))
    assert bar.symbol.canonical == canonical


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"[]",
        b"{}",
        b'{"data":[1]}',
    ],
)
def test_rejects_malformed_table_payload(body: bytes) -> None:
    with pytest.raises(AksharePayloadError):
        normalize_daily_bars(body)


@pytest.mark.parametrize(
    "overrides",
    [
        {"股票代码": "900001"},
        {"日期": "not-a-date"},
        {"开盘": "not-a-number"},
        {"成交量": "Infinity"},
        {"收盘": None},
    ],
)
def test_rejects_unsafe_rows(overrides: dict[str, object]) -> None:
    with pytest.raises((AksharePayloadError, ValueError)):
        normalize_daily_bars(_payload(**overrides))


def test_financial_invariants_are_enforced_by_domain_model() -> None:
    with pytest.raises(ValueError, match="low <="):
        normalize_daily_bars(_payload(**{"最低": 10.2, "开盘": 10.1}))
