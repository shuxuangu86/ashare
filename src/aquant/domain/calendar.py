from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from aquant.domain.enums import Exchange


@dataclass(frozen=True, slots=True)
class TradingSession:
    exchange: Exchange
    trade_date: date
    is_open: bool


class TradingCalendar:
    def __init__(self, sessions: Iterable[TradingSession]) -> None:
        by_key: dict[tuple[Exchange, date], TradingSession] = {}
        for session in sessions:
            key = (session.exchange, session.trade_date)
            if key in by_key:
                raise ValueError(
                    f"duplicate trading session: {session.exchange} {session.trade_date}"
                )
            by_key[key] = session
        self._by_key = by_key
        self._open_dates = {
            exchange: tuple(
                sorted(
                    session.trade_date
                    for session in by_key.values()
                    if session.exchange is exchange and session.is_open
                )
            )
            for exchange in Exchange
        }

    def session(self, exchange: Exchange, trade_date: date) -> TradingSession:
        try:
            return self._by_key[(exchange, trade_date)]
        except KeyError as exc:
            raise KeyError(f"calendar has no session for {exchange} on {trade_date}") from exc

    def is_open(self, exchange: Exchange, trade_date: date) -> bool:
        return self.session(exchange, trade_date).is_open

    def previous_open_date(self, exchange: Exchange, value: date) -> date:
        dates = self._open_dates[exchange]
        index = bisect_left(dates, value) - 1
        if index < 0:
            raise LookupError(f"no previous open date for {exchange} before {value}")
        return dates[index]

    def next_open_date(self, exchange: Exchange, value: date) -> date:
        dates = self._open_dates[exchange]
        index = bisect_right(dates, value)
        if index >= len(dates):
            raise LookupError(f"no next open date for {exchange} after {value}")
        return dates[index]

    def open_dates(self, exchange: Exchange) -> tuple[date, ...]:
        return self._open_dates[exchange]
