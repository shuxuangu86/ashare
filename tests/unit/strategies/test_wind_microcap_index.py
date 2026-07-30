from datetime import date
from decimal import Decimal

import pytest

from aquant.domain.enums import Exchange
from aquant.domain.identifiers import Symbol
from aquant.strategies.microcap import (
    WindMicrocapConstituentReturn,
    calculate_wind_microcap_daily_equal,
)

SYMBOLS = (
    Symbol("600001", Exchange.XSHG),
    Symbol("000001", Exchange.XSHE),
    Symbol("600002", Exchange.XSHG),
)


def _return(day: date, symbol: Symbol, value: str) -> WindMicrocapConstituentReturn:
    return WindMicrocapConstituentReturn(day, symbol, Decimal(value))


def test_daily_equal_index_uses_arithmetic_return_and_compounds() -> None:
    first = date(2024, 1, 2)
    second = date(2024, 1, 3)
    points = calculate_wind_microcap_daily_equal(
        (
            _return(first, SYMBOLS[0], "0.10"),
            _return(first, SYMBOLS[1], "-0.10"),
            _return(second, SYMBOLS[0], "0"),
            _return(second, SYMBOLS[1], "0.20"),
        ),
        target_count=2,
    )
    assert points[0].simple_return == 0
    assert points[0].net_value == 1
    assert points[0].two_way_turnover == 0
    assert points[1].simple_return == Decimal("0.10")
    assert points[1].net_value == Decimal("1.10")
    assert points[1].two_way_turnover == Decimal("0.10")
    assert points[1].one_way_turnover == Decimal("0.05")


def test_daily_equal_index_turnover_includes_constituent_replacement() -> None:
    first = date(2024, 1, 2)
    second = date(2024, 1, 3)
    points = calculate_wind_microcap_daily_equal(
        (
            _return(first, SYMBOLS[0], "0"),
            _return(first, SYMBOLS[1], "0"),
            _return(second, SYMBOLS[1], "0"),
            _return(second, SYMBOLS[2], "0"),
        ),
        target_count=2,
    )
    assert points[1].two_way_turnover == Decimal("1.0")
    assert points[1].one_way_turnover == Decimal("0.5")


def test_daily_equal_index_rejects_bad_or_incomplete_inputs() -> None:
    day = date(2024, 1, 2)
    with pytest.raises(ValueError, match="greater than"):
        _return(day, SYMBOLS[0], "-1")
    with pytest.raises(ValueError, match="must not be empty"):
        calculate_wind_microcap_daily_equal((), target_count=2)
    with pytest.raises(ValueError, match="incomplete"):
        calculate_wind_microcap_daily_equal(
            (_return(day, SYMBOLS[0], "0"),),
            target_count=2,
        )
    duplicate = _return(day, SYMBOLS[0], "0")
    with pytest.raises(ValueError, match="duplicate"):
        calculate_wind_microcap_daily_equal(
            (duplicate, duplicate),
            target_count=2,
        )
