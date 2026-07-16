from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum

from aquant.domain.identifiers import Symbol


class AdjustmentFactorType(StrEnum):
    CUMULATIVE = "CUMULATIVE"


class LimitStatus(StrEnum):
    NONE = "NONE"
    LIMIT_UP = "LIMIT_UP"
    LIMIT_DOWN = "LIMIT_DOWN"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class DailyBar:
    """Unadjusted daily OHLCV. Adjustment factors are stored separately."""

    symbol: Symbol
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    amount: Decimal

    def __post_init__(self) -> None:
        for field_name in ("open", "high", "low", "close", "volume", "amount"):
            value = Decimal(getattr(self, field_name))
            if not value.is_finite():
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("OHLC prices must be positive")
        if self.volume < 0 or self.amount < 0:
            raise ValueError("volume and amount must be non-negative")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("daily bar must satisfy low <= open/close <= high")
        if self.low > self.high:
            raise ValueError("low cannot exceed high")


@dataclass(frozen=True, slots=True)
class AdjustmentFactor:
    symbol: Symbol
    trade_date: date
    value: Decimal
    factor_type: AdjustmentFactorType = AdjustmentFactorType.CUMULATIVE

    def __post_init__(self) -> None:
        value = Decimal(self.value)
        if not value.is_finite() or value <= 0:
            raise ValueError("adjustment factor must be finite and positive")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class SecurityStatus:
    symbol: Symbol
    trade_date: date
    suspended: bool
    is_st: bool
    limit_status: LimitStatus = LimitStatus.UNKNOWN

    @property
    def can_buy(self) -> bool:
        return not self.suspended and self.limit_status is not LimitStatus.LIMIT_UP

    @property
    def can_sell(self) -> bool:
        return not self.suspended and self.limit_status is not LimitStatus.LIMIT_DOWN
