from datetime import date

import pytest

from aquant.domain.calendar import TradingCalendar, TradingSession
from aquant.domain.enums import Board, Exchange
from aquant.domain.identifiers import Symbol
from aquant.domain.instruments import Instrument


def test_instrument_listing_interval_is_inclusive() -> None:
    instrument = Instrument(
        symbol=Symbol("600000", Exchange.XSHG),
        name=" 浦发银行 ",
        list_date=date(1999, 11, 10),
        delist_date=date(2026, 7, 16),
        board=Board.MAIN,
    )

    assert instrument.name == "浦发银行"
    assert instrument.is_listed_on(date(1999, 11, 10))
    assert instrument.is_listed_on(date(2026, 7, 16))
    assert not instrument.is_listed_on(date(2026, 7, 17))


def test_active_instrument_has_open_ended_listing() -> None:
    instrument = Instrument(
        symbol=Symbol("000001", Exchange.XSHE),
        name="平安银行",
        list_date=date(1991, 4, 3),
    )

    assert instrument.is_listed_on(date(2099, 1, 1))


def test_instrument_rejects_invalid_dates_and_name() -> None:
    with pytest.raises(ValueError, match="name"):
        Instrument(Symbol("600000", Exchange.XSHG), " ", date(2020, 1, 1))
    with pytest.raises(ValueError, match="delist_date"):
        Instrument(
            Symbol("600000", Exchange.XSHG),
            "浦发银行",
            date(2020, 1, 2),
            date(2020, 1, 1),
        )


def _calendar() -> TradingCalendar:
    return TradingCalendar(
        [
            TradingSession(Exchange.XSHG, date(2026, 7, 13), True),
            TradingSession(Exchange.XSHG, date(2026, 7, 14), True),
            TradingSession(Exchange.XSHG, date(2026, 7, 15), False),
            TradingSession(Exchange.XSHG, date(2026, 7, 16), True),
            TradingSession(Exchange.XSHE, date(2026, 7, 16), True),
        ]
    )


def test_trading_calendar_navigation_skips_closed_sessions() -> None:
    calendar = _calendar()

    assert calendar.session(Exchange.XSHG, date(2026, 7, 15)).is_open is False
    assert calendar.is_open(Exchange.XSHG, date(2026, 7, 16))
    assert calendar.previous_open_date(Exchange.XSHG, date(2026, 7, 16)) == date(2026, 7, 14)
    assert calendar.next_open_date(Exchange.XSHG, date(2026, 7, 14)) == date(2026, 7, 16)
    assert calendar.open_dates(Exchange.XSHE) == (date(2026, 7, 16),)


def test_trading_calendar_rejects_duplicates() -> None:
    session = TradingSession(Exchange.XSHG, date(2026, 7, 16), True)
    with pytest.raises(ValueError, match="duplicate"):
        TradingCalendar([session, session])


def test_trading_calendar_rejects_unknown_and_out_of_range_navigation() -> None:
    calendar = _calendar()
    with pytest.raises(KeyError, match="no session"):
        calendar.session(Exchange.XSHE, date(2026, 7, 15))
    with pytest.raises(LookupError, match="no previous"):
        calendar.previous_open_date(Exchange.XSHG, date(2026, 7, 13))
    with pytest.raises(LookupError, match="no next"):
        calendar.next_open_date(Exchange.XSHE, date(2026, 7, 16))
