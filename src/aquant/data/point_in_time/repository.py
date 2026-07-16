from dataclasses import dataclass
from datetime import date, datetime

from aquant.data.point_in_time.table import PointInTimeRecord, PointInTimeTable
from aquant.domain.fundamentals import FinancialStatement, IndexMembership, StatementType
from aquant.domain.identifiers import Symbol


@dataclass(frozen=True, slots=True)
class FinancialStatementKey:
    symbol: Symbol
    period_end: date
    statement_type: StatementType


class PointInTimeRepository:
    """Typed PIT queries; every read requires an explicit knowledge time."""

    def __init__(
        self,
        *,
        financials: PointInTimeTable[FinancialStatementKey, FinancialStatement],
        index_memberships: PointInTimeTable[tuple[str, Symbol], IndexMembership],
    ) -> None:
        self._financials = financials
        self._index_memberships = index_memberships

    def financial_statement(
        self,
        symbol: Symbol,
        period_end: date,
        statement_type: StatementType,
        *,
        asof_time: datetime,
    ) -> PointInTimeRecord[FinancialStatementKey, FinancialStatement] | None:
        return self._financials.latest(
            FinancialStatementKey(symbol, period_end, statement_type), asof_time
        )

    def index_constituents(
        self,
        index_code: str,
        *,
        trade_date: date,
        asof_time: datetime,
    ) -> tuple[Symbol, ...]:
        resolved_index = index_code.strip().upper()
        visible = self._index_memberships.latest_by_key(asof_time)
        symbols = {
            membership.value.symbol
            for membership in visible.values()
            if membership.value.index_code == resolved_index
            and membership.value.contains(trade_date)
        }
        return tuple(sorted(symbols))
