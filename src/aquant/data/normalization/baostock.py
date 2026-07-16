import json
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar


class BaoStockPayloadError(ValueError):
    pass


def normalize_baostock_daily_bars(body: bytes) -> tuple[DailyBar, ...]:
    fields, items = _payload(body)
    required = {"date", "code", "open", "high", "low", "close", "volume", "amount"}
    missing = required.difference(fields)
    if missing:
        raise BaoStockPayloadError(f"BaoStock payload missing fields: {sorted(missing)}")
    bars: list[DailyBar] = []
    for item in items:
        if len(item) != len(fields):
            raise BaoStockPayloadError("BaoStock item length does not match fields")
        row = dict(zip(fields, item, strict=True))
        try:
            symbol = _symbol(row["code"])
            trade_date = date.fromisoformat(row["date"])
        except ValueError as exc:
            raise BaoStockPayloadError("invalid BaoStock identifier or date") from exc
        bars.append(
            DailyBar(
                symbol=symbol,
                trade_date=trade_date,
                open=_decimal(row["open"], "open"),
                high=_decimal(row["high"], "high"),
                low=_decimal(row["low"], "low"),
                close=_decimal(row["close"], "close"),
                volume=_decimal(row["volume"], "volume"),
                amount=_decimal(row["amount"], "amount"),
            )
        )
    return tuple(bars)


def _payload(body: bytes) -> tuple[list[str], list[list[str]]]:
    try:
        payload: Any = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaoStockPayloadError("BaoStock payload is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise BaoStockPayloadError("BaoStock payload root must be an object")
    fields = payload.get("fields")
    items = payload.get("items")
    if not isinstance(fields, list) or not all(isinstance(value, str) for value in fields):
        raise BaoStockPayloadError("BaoStock fields must be strings")
    if not isinstance(items, list) or not all(
        isinstance(item, list) and all(isinstance(value, str) for value in item) for item in items
    ):
        raise BaoStockPayloadError("BaoStock items must be string arrays")
    return fields, items


def _symbol(value: str) -> Symbol:
    try:
        venue, code = value.split(".", maxsplit=1)
    except ValueError as exc:
        raise BaoStockPayloadError(f"invalid BaoStock code: {value}") from exc
    exchange = {"sh": "XSHG", "sz": "XSHE"}.get(venue.lower())
    if exchange is None:
        raise BaoStockPayloadError(f"unsupported BaoStock venue: {venue}")
    return Symbol.parse(f"{code}.{exchange}")


def _decimal(value: str, field: str) -> Decimal:
    try:
        resolved = Decimal(value)
    except InvalidOperation as exc:
        raise BaoStockPayloadError(f"invalid BaoStock numeric field {field}") from exc
    if not resolved.is_finite():
        raise BaoStockPayloadError(f"invalid BaoStock numeric field {field}")
    return resolved
