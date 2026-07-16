from datetime import UTC, date, datetime
from decimal import Decimal

from aquant.data.point_in_time import (
    FinancialStatementKey,
    PointInTimeRecord,
    PointInTimeRepository,
    PointInTimeTable,
)
from aquant.domain.fundamentals import FinancialStatement, IndexMembership, StatementType
from aquant.domain.identifiers import Symbol

SYMBOL = Symbol.parse("600000.XSHG")
PERIOD_END = date(2025, 12, 31)


def _time(month: int, day: int) -> datetime:
    return datetime(2026, month, day, 18, tzinfo=UTC)


def _repository() -> PointInTimeRepository:
    key = FinancialStatementKey(SYMBOL, PERIOD_END, StatementType.INCOME_STATEMENT)
    original = FinancialStatement(
        SYMBOL,
        PERIOD_END,
        StatementType.INCOME_STATEMENT,
        (("net_income", Decimal("100")),),
    )
    revised = FinancialStatement(
        SYMBOL,
        PERIOD_END,
        StatementType.INCOME_STATEMENT,
        (("net_income", Decimal("90")),),
    )
    financials = PointInTimeTable(
        [
            PointInTimeRecord(
                key,
                original,
                available_at=_time(3, 20),
                announcement_date=_time(3, 20),
                ingested_at=_time(3, 21),
                revision_no=0,
                source="akshare",
                source_record_id="f1",
                period_end=PERIOD_END,
            ),
            PointInTimeRecord(
                key,
                revised,
                available_at=_time(4, 10),
                announcement_date=_time(4, 10),
                ingested_at=_time(4, 11),
                revision_no=1,
                source="akshare",
                source_record_id="f2",
                period_end=PERIOD_END,
            ),
        ]
    )
    membership = IndexMembership("000300.XSHG", SYMBOL, date(2026, 6, 15))
    memberships = PointInTimeTable(
        [
            PointInTimeRecord(
                (membership.index_code, SYMBOL),
                membership,
                available_at=_time(6, 12),
                ingested_at=_time(6, 12),
                revision_no=0,
                source="akshare",
                source_record_id="m1",
            )
        ]
    )
    return PointInTimeRepository(financials=financials, index_memberships=memberships)


def test_financial_statement_is_invisible_before_announcement() -> None:
    record = _repository().financial_statement(
        SYMBOL,
        PERIOD_END,
        StatementType.INCOME_STATEMENT,
        asof_time=_time(3, 19),
    )
    assert record is None


def test_financial_revision_only_replaces_value_after_available_at() -> None:
    repository = _repository()
    original = repository.financial_statement(
        SYMBOL, PERIOD_END, StatementType.INCOME_STATEMENT, asof_time=_time(4, 1)
    )
    revised = repository.financial_statement(
        SYMBOL, PERIOD_END, StatementType.INCOME_STATEMENT, asof_time=_time(4, 12)
    )
    assert original is not None and original.value.metric("net_income") == Decimal("100")
    assert revised is not None and revised.value.metric("net_income") == Decimal("90")


def test_index_pool_requires_both_known_and_effective_membership() -> None:
    repository = _repository()
    assert (
        repository.index_constituents(
            "000300.XSHG", trade_date=date(2026, 6, 11), asof_time=_time(6, 16)
        )
        == ()
    )
    assert (
        repository.index_constituents(
            "000300.XSHG", trade_date=date(2026, 6, 16), asof_time=_time(6, 11)
        )
        == ()
    )
    assert repository.index_constituents(
        "000300.XSHG", trade_date=date(2026, 6, 16), asof_time=_time(6, 16)
    ) == (SYMBOL,)
