from datetime import datetime
from decimal import Decimal
from uuid import UUID

from aquant.backtest.matching.orders import BacktestOrder, Fill
from aquant.domain.market_data import DailyBar, SecurityStatus
from aquant.domain.time import require_aware


class NextOpenMatcher:
    """Day 9 baseline: fills at a later session open without costs or market constraints."""

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
        del available_cash, status
        timestamp = require_aware(occurred_at, field_name="occurred_at")
        if bar.symbol != order.symbol:
            raise ValueError("matching bar symbol does not match order")
        if timestamp <= order.submitted_at:
            raise ValueError("next-open match must occur after order submission")
        quantity = order.remaining_quantity
        if maximum_quantity is not None:
            if maximum_quantity < 0:
                raise ValueError("maximum fill quantity cannot be negative")
            quantity = min(quantity, maximum_quantity)
        if quantity == 0:
            return None
        return Fill(
            fill_id=fill_id,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=bar.open,
            fee=Decimal("0"),
            occurred_at=timestamp,
        )
