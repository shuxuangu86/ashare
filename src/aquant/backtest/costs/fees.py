from dataclasses import dataclass
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
