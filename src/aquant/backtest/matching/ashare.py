from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_FLOOR, Decimal
from uuid import UUID

from aquant.backtest.costs import AshareFeeModel
from aquant.backtest.matching.orders import BacktestOrder, Fill
from aquant.domain.enums import Side
from aquant.domain.market_data import DailyBar, LimitStatus, SecurityStatus
from aquant.domain.time import require_aware


@dataclass(frozen=True, slots=True)
class AshareExecutionRules:
    lot_size: int = 100
    max_volume_participation: Decimal = Decimal("0.10")
    slippage_bps: Decimal = Decimal("5")
    require_security_status: bool = True

    def __post_init__(self) -> None:
        if self.lot_size <= 0:
            raise ValueError("lot size must be positive")
        participation = Decimal(self.max_volume_participation)
        slippage = Decimal(self.slippage_bps)
        if not Decimal("0") < participation <= Decimal("1"):
            raise ValueError("volume participation must be in (0, 1]")
        if not slippage.is_finite() or slippage < 0:
            raise ValueError("slippage must be finite and non-negative")


class AshareOpenMatcher:
    """Rule-aware A-share daily matcher used by the Day 10 backtest path."""

    def __init__(
        self,
        *,
        rules: AshareExecutionRules | None = None,
        fee_model: AshareFeeModel | None = None,
    ) -> None:
        self._rules = rules or AshareExecutionRules()
        self._fee_model = fee_model or AshareFeeModel()

    def match(
        self,
        order: BacktestOrder,
        bar: DailyBar,
        *,
        occurred_at: datetime,
        fill_id: UUID,
        maximum_quantity: int | None = None,
        available_cash: Decimal | None = None,
        status: SecurityStatus | None = None,
    ) -> Fill | None:
        timestamp = require_aware(occurred_at, field_name="occurred_at")
        if bar.symbol != order.symbol or (status is not None and status.symbol != order.symbol):
            raise ValueError("market state symbol does not match order")
        if timestamp <= order.submitted_at:
            raise ValueError("next-open match must occur after order submission")
        if self._rules.require_security_status and status is None:
            return None
        if status is not None and (
            status.suspended
            or (order.side is Side.BUY and status.limit_status is LimitStatus.LIMIT_UP)
            or (order.side is Side.SELL and status.limit_status is LimitStatus.LIMIT_DOWN)
        ):
            return None

        quantity = order.remaining_quantity
        if maximum_quantity is not None:
            if maximum_quantity < 0:
                raise ValueError("maximum fill quantity cannot be negative")
            quantity = min(quantity, maximum_quantity)
        volume_cap = int(
            (bar.volume * self._rules.max_volume_participation).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
        quantity = min(quantity, volume_cap)
        quantity = quantity // self._rules.lot_size * self._rules.lot_size
        if quantity == 0:
            return None

        slip = self._rules.slippage_bps / Decimal("10000")
        price = bar.open * (Decimal("1") + slip if order.side is Side.BUY else Decimal("1") - slip)
        if price <= 0:
            raise ValueError("slippage produced a non-positive execution price")
        if order.side is Side.BUY and available_cash is not None:
            quantity = self._affordable_quantity(
                quantity, price, Decimal(available_cash), order.side
            )
            if quantity == 0:
                return None
        fee = self._fee_model.calculate(order.side, price * quantity).total
        return Fill(
            fill_id,
            order.order_id,
            order.symbol,
            order.side,
            quantity,
            price,
            fee,
            timestamp,
        )

    def _affordable_quantity(
        self, requested: int, price: Decimal, cash: Decimal, side: Side
    ) -> int:
        quantity = requested
        while quantity > 0:
            notional = price * quantity
            if notional + self._fee_model.calculate(side, notional).total <= cash:
                return quantity
            quantity -= self._rules.lot_size
        return 0
