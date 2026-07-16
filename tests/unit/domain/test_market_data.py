from datetime import date
from decimal import Decimal

import pytest

from aquant.domain.enums import Exchange
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import AdjustmentFactor, DailyBar, LimitStatus, SecurityStatus

SYMBOL = Symbol("600000", Exchange.XSHG)


def _bar(**overrides: object) -> DailyBar:
    values: dict[str, object] = {
        "symbol": SYMBOL,
        "trade_date": date(2026, 7, 16),
        "open": Decimal("10.00"),
        "high": Decimal("10.50"),
        "low": Decimal("9.80"),
        "close": Decimal("10.20"),
        "volume": Decimal("100000"),
        "amount": Decimal("1020000"),
    }
    values.update(overrides)
    return DailyBar(**values)  # type: ignore[arg-type]


def test_unadjusted_daily_bar_preserves_decimal_values() -> None:
    bar = _bar()
    assert bar.close == Decimal("10.20")
    assert bar.volume == Decimal("100000")


@pytest.mark.parametrize(
    "overrides",
    [
        {"low": Decimal("10.10")},
        {"high": Decimal("10.10")},
        {"low": Decimal("11"), "high": Decimal("10")},
        {"open": Decimal("0")},
        {"volume": Decimal("-1")},
        {"amount": Decimal("-1")},
        {"close": Decimal("NaN")},
    ],
)
def test_daily_bar_rejects_financially_impossible_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _bar(**overrides)


@pytest.mark.parametrize("value", [Decimal("0"), Decimal("-1"), Decimal("Infinity")])
def test_adjustment_factor_must_be_positive_and_finite(value: Decimal) -> None:
    with pytest.raises(ValueError):
        AdjustmentFactor(SYMBOL, date(2026, 7, 16), value)


def test_adjustment_factor_keeps_unadjusted_prices_separate() -> None:
    factor = AdjustmentFactor(SYMBOL, date(2026, 7, 16), Decimal("1.2345"))
    assert factor.value == Decimal("1.2345")


@pytest.mark.parametrize(
    ("status", "can_buy", "can_sell"),
    [
        (SecurityStatus(SYMBOL, date(2026, 7, 16), False, False, LimitStatus.NONE), True, True),
        (
            SecurityStatus(SYMBOL, date(2026, 7, 16), False, False, LimitStatus.LIMIT_UP),
            False,
            True,
        ),
        (
            SecurityStatus(SYMBOL, date(2026, 7, 16), False, False, LimitStatus.LIMIT_DOWN),
            True,
            False,
        ),
        (
            SecurityStatus(SYMBOL, date(2026, 7, 16), True, False, LimitStatus.NONE),
            False,
            False,
        ),
    ],
)
def test_security_status_controls_trade_direction(
    status: SecurityStatus, can_buy: bool, can_sell: bool
) -> None:
    assert status.can_buy is can_buy
    assert status.can_sell is can_sell
