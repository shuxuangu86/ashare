import json
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from aquant.domain.enums import Exchange
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar


class AksharePayloadError(ValueError):
    pass


def normalize_daily_bars(body: bytes) -> tuple[DailyBar, ...]:
    records = _table_records(body)
    return tuple(
        DailyBar(
            symbol=_symbol(_required(record, "股票代码")),
            trade_date=_date(_required(record, "日期")),
            open=_decimal(_required(record, "开盘"), field="开盘"),
            high=_decimal(_required(record, "最高"), field="最高"),
            low=_decimal(_required(record, "最低"), field="最低"),
            close=_decimal(_required(record, "收盘"), field="收盘"),
            volume=_decimal(_required(record, "成交量"), field="成交量") * Decimal("100"),
            amount=_decimal(_required(record, "成交额"), field="成交额"),
        )
        for record in records
    )


def _table_records(body: bytes) -> tuple[dict[str, Any], ...]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AksharePayloadError("AKShare table is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise AksharePayloadError("AKShare table root must be an object")
    records = payload.get("data")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise AksharePayloadError("AKShare table data must be a list of objects")
    return tuple(records)


def _required(record: dict[str, Any], field: str) -> object:
    value = record.get(field)
    if value is None or value == "":
        raise AksharePayloadError(f"AKShare row is missing required field: {field}")
    return value


def _symbol(value: object) -> Symbol:
    code = str(value).strip().zfill(6)
    xshg_prefixes = ("600", "601", "603", "605", "688", "689")
    xshe_prefixes = ("000", "001", "002", "003", "300", "301")
    if code.startswith(xshg_prefixes):
        return Symbol(code, Exchange.XSHG)
    if code.startswith(xshe_prefixes):
        return Symbol(code, Exchange.XSHE)
    raise AksharePayloadError(f"unsupported A-share code: {code!r}")


def _date(value: object) -> date:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise AksharePayloadError(f"invalid AKShare trade date: {value!r}") from exc


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        resolved = Decimal(str(value))
    except InvalidOperation as exc:
        raise AksharePayloadError(f"invalid AKShare numeric field {field}: {value!r}") from exc
    if not resolved.is_finite():
        raise AksharePayloadError(f"invalid AKShare numeric field {field}: {value!r}")
    return resolved
