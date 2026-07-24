from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from aquant.domain.enums import Side


@dataclass(frozen=True, slots=True)
class FeeBreakdown:
    commission: Decimal
    stamp_duty: Decimal
    transfer_fee: Decimal

    @property
    def total(self) -> Decimal:
        return self.commission + self.stamp_duty + self.transfer_fee


@dataclass(frozen=True, slots=True)
class AshareFeeModel:
    commission_rate: Decimal = Decimal("0.0003")
    minimum_commission: Decimal = Decimal("5")
    sell_stamp_duty_rate: Decimal = Decimal("0.0005")
    transfer_fee_rate: Decimal = Decimal("0.00001")

    def __post_init__(self) -> None:
        values = (
            self.commission_rate,
            self.minimum_commission,
            self.sell_stamp_duty_rate,
            self.transfer_fee_rate,
        )
        if any(not Decimal(value).is_finite() or Decimal(value) < 0 for value in values):
            raise ValueError("fee model parameters must be finite and non-negative")

    def calculate(self, side: Side, notional: Decimal) -> FeeBreakdown:
        amount = Decimal(notional)
        if not amount.is_finite() or amount <= 0:
            raise ValueError("fee notional must be finite and positive")
        commission = max(amount * self.commission_rate, self.minimum_commission)
        stamp = amount * self.sell_stamp_duty_rate if side is Side.SELL else Decimal("0")
        transfer = amount * self.transfer_fee_rate
        return FeeBreakdown(commission, stamp, transfer)


@dataclass(frozen=True, slots=True)
class AshareFeeSchedule:
    """Date-effective A-share fee schedule supported from 2015-07-09."""

    commission_rate: Decimal = Decimal("0.0003")
    minimum_commission: Decimal = Decimal("5")
    cost_multiplier: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        values = (self.commission_rate, self.minimum_commission, self.cost_multiplier)
        if any(not Decimal(value).is_finite() or Decimal(value) < 0 for value in values):
            raise ValueError("fee schedule parameters must be finite and non-negative")

    def model_for(self, trade_date: date) -> AshareFeeModel:
        if trade_date < date(2015, 7, 9):
            raise ValueError("A-share fee schedule is unsupported before 2015-07-09")
        transfer_fee = Decimal("0.00001") if trade_date >= date(2022, 4, 29) else Decimal("0.00002")
        stamp_duty = Decimal("0.0005") if trade_date >= date(2023, 8, 28) else Decimal("0.001")
        multiplier = self.cost_multiplier
        return AshareFeeModel(
            commission_rate=self.commission_rate * multiplier,
            minimum_commission=self.minimum_commission * multiplier,
            sell_stamp_duty_rate=stamp_duty * multiplier,
            transfer_fee_rate=transfer_fee * multiplier,
        )
