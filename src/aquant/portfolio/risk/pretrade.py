from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from aquant.domain.enums import Side
from aquant.domain.market_data import SecurityStatus
from aquant.domain.time import require_aware
from aquant.execution.broker_api import BrokerAccountSnapshot, BrokerOrderRequest


@dataclass(frozen=True, slots=True)
class PreTradeRiskLimits:
    maximum_order_notional: Decimal = Decimal("100000")
    maximum_volume_participation: Decimal = Decimal("0.10")
    maximum_market_data_age: timedelta = timedelta(seconds=30)
    forbid_st: bool = True


@dataclass(frozen=True, slots=True)
class PreTradeContext:
    now: datetime
    market_data_at: datetime
    price: Decimal
    daily_volume: Decimal
    security_status: SecurityStatus
    account: BrokerAccountSnapshot
    data_release_published: bool


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    reasons: tuple[str, ...]


class PreTradeRiskEngine:
    def __init__(self, limits: PreTradeRiskLimits | None = None) -> None:
        self._limits = limits or PreTradeRiskLimits()

    def check(self, request: BrokerOrderRequest, context: PreTradeContext) -> RiskDecision:
        now = require_aware(context.now, field_name="now")
        market_at = require_aware(context.market_data_at, field_name="market_data_at")
        reasons: list[str] = []
        notional = context.price * request.quantity
        if not context.data_release_published:
            reasons.append("DATA_RELEASE_UNPUBLISHED")
        if now - market_at > self._limits.maximum_market_data_age or market_at > now:
            reasons.append("MARKET_DATA_STALE")
        if context.security_status.suspended:
            reasons.append("SECURITY_SUSPENDED")
        if self._limits.forbid_st and context.security_status.is_st:
            reasons.append("SECURITY_ST")
        if notional > self._limits.maximum_order_notional:
            reasons.append("ORDER_NOTIONAL_LIMIT")
        if context.daily_volume <= 0 or Decimal(request.quantity) / context.daily_volume > (
            self._limits.maximum_volume_participation
        ):
            reasons.append("VOLUME_PARTICIPATION_LIMIT")
        if request.side is Side.BUY and notional > context.account.cash:
            reasons.append("INSUFFICIENT_CASH")
        if request.side is Side.SELL and request.quantity > context.account.positions.get(
            request.symbol, 0
        ):
            reasons.append("INSUFFICIENT_POSITION")
        return RiskDecision(not reasons, tuple(reasons))
