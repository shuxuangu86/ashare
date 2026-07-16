import json
from datetime import date, datetime
from typing import Any

from aquant.domain.calendar import TradingSession
from aquant.domain.enums import Board, Exchange
from aquant.domain.identifiers import Symbol
from aquant.domain.instruments import Instrument


class TusharePayloadError(ValueError):
    pass


def parse_tabular_payload(body: bytes) -> tuple[dict[str, Any], ...]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TusharePayloadError("Tushare response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise TusharePayloadError("Tushare response root must be an object")
    code = payload.get("code")
    if code != 0:
        message = payload.get("msg")
        raise TusharePayloadError(f"Tushare API error {code}: {message}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise TusharePayloadError("Tushare response data must be an object")
    fields = data.get("fields")
    items = data.get("items")
    if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
        raise TusharePayloadError("Tushare fields must be a list of strings")
    if not isinstance(items, list):
        raise TusharePayloadError("Tushare items must be a list")

    records: list[dict[str, Any]] = []
    for row_number, item in enumerate(items):
        if not isinstance(item, list) or len(item) != len(fields):
            raise TusharePayloadError(f"Tushare row {row_number} does not match fields")
        records.append(dict(zip(fields, item, strict=True)))
    return tuple(records)


def normalize_instruments(body: bytes) -> tuple[Instrument, ...]:
    instruments: list[Instrument] = []
    for record in parse_tabular_payload(body):
        exchange = _exchange(record.get("exchange"))
        market = _required_text(record, "market")
        instruments.append(
            Instrument(
                symbol=Symbol(_required_text(record, "symbol"), exchange),
                name=_required_text(record, "name"),
                list_date=_date_yyyymmdd(record.get("list_date"), field="list_date"),
                delist_date=_optional_date_yyyymmdd(record.get("delist_date"), field="delist_date"),
                board=_board(market),
            )
        )
    return tuple(instruments)


def normalize_trading_calendar(body: bytes) -> tuple[TradingSession, ...]:
    sessions: list[TradingSession] = []
    for record in parse_tabular_payload(body):
        raw_is_open = record.get("is_open")
        if str(raw_is_open) not in {"0", "1"}:
            raise TusharePayloadError(f"invalid is_open value: {raw_is_open!r}")
        sessions.append(
            TradingSession(
                exchange=_exchange(record.get("exchange")),
                trade_date=_date_yyyymmdd(record.get("cal_date"), field="cal_date"),
                is_open=str(raw_is_open) == "1",
            )
        )
    return tuple(sessions)


def _required_text(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise TusharePayloadError(f"missing required text field: {field}")
    return value.strip()


def _exchange(value: object) -> Exchange:
    mapping = {"SSE": Exchange.XSHG, "SZSE": Exchange.XSHE}
    try:
        return mapping[str(value).upper()]
    except KeyError as exc:
        raise TusharePayloadError(f"unsupported exchange: {value!r}") from exc


def _board(value: str) -> Board:
    mapping = {
        "主板": Board.MAIN,
        "MAIN BOARD": Board.MAIN,
        "科创板": Board.STAR,
        "STAR MARKET": Board.STAR,
        "创业板": Board.CHINEXT,
        "CHINEXT": Board.CHINEXT,
    }
    return mapping.get(value.strip().upper(), Board.OTHER)


def _date_yyyymmdd(value: object, *, field: str) -> date:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise TusharePayloadError(f"invalid {field}: {value!r}")
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise TusharePayloadError(f"invalid {field}: {value!r}") from exc


def _optional_date_yyyymmdd(value: object, *, field: str) -> date | None:
    if value is None or value == "":
        return None
    return _date_yyyymmdd(value, field=field)
