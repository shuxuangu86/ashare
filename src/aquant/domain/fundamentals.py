from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum

from aquant.domain.identifiers import Symbol


class StatementType(StrEnum):
    BALANCE_SHEET = "BALANCE_SHEET"
    INCOME_STATEMENT = "INCOME_STATEMENT"
    CASH_FLOW = "CASH_FLOW"


@dataclass(frozen=True, slots=True)
class FinancialStatement:
    symbol: Symbol
    period_end: date
    statement_type: StatementType
    metrics: tuple[tuple[str, Decimal], ...]

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError("financial statement metrics must not be empty")
        normalized: list[tuple[str, Decimal]] = []
        names: set[str] = set()
        for name, raw_value in self.metrics:
            resolved_name = name.strip().lower()
            value = Decimal(raw_value)
            if not resolved_name or resolved_name in names:
                raise ValueError("financial metric names must be unique and non-blank")
            if not value.is_finite():
                raise ValueError("financial metric values must be finite")
            names.add(resolved_name)
            normalized.append((resolved_name, value))
        object.__setattr__(self, "metrics", tuple(sorted(normalized)))

    def metric(self, name: str) -> Decimal:
        resolved = name.strip().lower()
        try:
            return dict(self.metrics)[resolved]
        except KeyError as exc:
            raise KeyError(f"financial metric is unavailable: {name}") from exc


@dataclass(frozen=True, slots=True)
class IndexMembership:
    index_code: str
    symbol: Symbol
    effective_from: date
    effective_to: date | None = None

    def __post_init__(self) -> None:
        index_code = self.index_code.strip().upper()
        if not index_code:
            raise ValueError("index_code must not be blank")
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("effective_to cannot be earlier than effective_from")
        object.__setattr__(self, "index_code", index_code)

    def contains(self, trade_date: date) -> bool:
        return self.effective_from <= trade_date and (
            self.effective_to is None or trade_date <= self.effective_to
        )


@dataclass(frozen=True, slots=True)
class IndustryMembership:
    taxonomy: str
    industry_code: str
    symbol: Symbol
    effective_from: date
    effective_to: date | None = None

    def __post_init__(self) -> None:
        if not self.taxonomy.strip() or not self.industry_code.strip():
            raise ValueError("taxonomy and industry_code must not be blank")
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("effective_to cannot be earlier than effective_from")
        object.__setattr__(self, "taxonomy", self.taxonomy.strip().upper())
        object.__setattr__(self, "industry_code", self.industry_code.strip().upper())

    def contains(self, trade_date: date) -> bool:
        return self.effective_from <= trade_date and (
            self.effective_to is None or trade_date <= self.effective_to
        )
