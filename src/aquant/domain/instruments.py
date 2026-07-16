from dataclasses import dataclass
from datetime import date

from aquant.domain.enums import Board, SecurityType
from aquant.domain.identifiers import Symbol


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: Symbol
    name: str
    list_date: date
    delist_date: date | None = None
    security_type: SecurityType = SecurityType.STOCK
    board: Board = Board.OTHER

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("instrument name must not be blank")
        if self.delist_date is not None and self.delist_date < self.list_date:
            raise ValueError("delist_date cannot be earlier than list_date")
        object.__setattr__(self, "name", self.name.strip())

    def is_listed_on(self, value: date) -> bool:
        return self.list_date <= value and (self.delist_date is None or value <= self.delist_date)
