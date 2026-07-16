from datetime import date
from decimal import Decimal

import pytest

from aquant.domain.fundamentals import (
    FinancialStatement,
    IndexMembership,
    IndustryMembership,
    StatementType,
)
from aquant.domain.identifiers import Symbol

SYMBOL = Symbol.parse("000001.XSHE")


def test_financial_metrics_are_canonical_and_queryable() -> None:
    statement = FinancialStatement(
        SYMBOL,
        date(2025, 12, 31),
        StatementType.BALANCE_SHEET,
        ((" Assets ", Decimal("10")), ("liabilities", Decimal("4"))),
    )
    assert statement.metric("ASSETS") == Decimal("10")
    with pytest.raises(KeyError):
        statement.metric("cash")


@pytest.mark.parametrize(
    "metrics",
    [(), (("", Decimal("1")),), (("x", Decimal("1")), ("X", Decimal("2")))],
)
def test_financial_metrics_reject_invalid_shapes(
    metrics: tuple[tuple[str, Decimal], ...],
) -> None:
    with pytest.raises(ValueError):
        FinancialStatement(SYMBOL, date(2025, 12, 31), StatementType.CASH_FLOW, metrics)


def test_membership_intervals_are_inclusive() -> None:
    index = IndexMembership(" 000300.xshg ", SYMBOL, date(2026, 1, 1), date(2026, 6, 30))
    industry = IndustryMembership("sw", "bank", SYMBOL, date(2026, 1, 1))
    assert index.index_code == "000300.XSHG"
    assert index.contains(date(2026, 6, 30))
    assert not index.contains(date(2026, 7, 1))
    assert industry.taxonomy == "SW" and industry.contains(date(2026, 7, 1))
