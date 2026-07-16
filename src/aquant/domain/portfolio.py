from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from aquant.domain.identifiers import Symbol
from aquant.domain.time import require_aware


@dataclass(frozen=True, slots=True)
class TargetPosition:
    symbol: Symbol
    target_weight: Decimal

    def __post_init__(self) -> None:
        weight = Decimal(self.target_weight)
        if not Decimal("0") <= weight <= Decimal("1"):
            raise ValueError("target_weight must be between 0 and 1")
        object.__setattr__(self, "target_weight", weight)


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    strategy_id: str
    trade_date: date
    asof_time: datetime
    data_release_id: str
    signal_version: str
    positions: tuple[TargetPosition, ...]

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id must not be blank")
        if not self.data_release_id.strip():
            raise ValueError("data_release_id must not be blank")
        if not self.signal_version.strip():
            raise ValueError("signal_version must not be blank")

        asof_time = require_aware(self.asof_time, field_name="asof_time")
        symbols = [position.symbol for position in self.positions]
        if len(symbols) != len(set(symbols)):
            raise ValueError("target portfolio cannot contain duplicate symbols")
        total_weight = sum(
            (position.target_weight for position in self.positions), start=Decimal("0")
        )
        if total_weight > Decimal("1"):
            raise ValueError("total target weight cannot exceed 1")
        object.__setattr__(self, "strategy_id", self.strategy_id.strip())
        object.__setattr__(self, "data_release_id", self.data_release_id.strip())
        object.__setattr__(self, "signal_version", self.signal_version.strip())
        object.__setattr__(self, "asof_time", asof_time)

    @property
    def cash_weight(self) -> Decimal:
        invested = sum((position.target_weight for position in self.positions), start=Decimal("0"))
        return Decimal("1") - invested
