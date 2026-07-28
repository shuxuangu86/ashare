from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from aquant.domain.enums import Exchange, Side
from aquant.domain.market_data import DailyBar, LimitStatus, SecurityStatus


class TradabilityReason(StrEnum):
    MISSING_MARKET_DATA = "MISSING_MARKET_DATA"
    MISSING_STATUS = "MISSING_STATUS"
    NOT_LISTED = "NOT_LISTED"
    DELISTED = "DELISTED"
    SUSPENDED = "SUSPENDED"
    ST = "ST"
    LIMIT_UP_BUY = "LIMIT_UP_BUY"
    LIMIT_DOWN_SELL = "LIMIT_DOWN_SELL"
    ONE_PRICE_BOARD = "ONE_PRICE_BOARD"
    INSUFFICIENT_AMOUNT = "INSUFFICIENT_AMOUNT"
    INSUFFICIENT_LISTING_AGE = "INSUFFICIENT_LISTING_AGE"
    OUTSIDE_MARKET_SCOPE = "OUTSIDE_MARKET_SCOPE"


@dataclass(frozen=True, slots=True)
class TradabilityRules:
    exclude_st: bool = True
    minimum_amount: Decimal = Decimal("0")
    minimum_listing_days: int = 0
    allowed_exchanges: tuple[Exchange, ...] = (
        Exchange.XSHG,
        Exchange.XSHE,
        Exchange.XBSE,
    )

    def __post_init__(self) -> None:
        amount = Decimal(self.minimum_amount)
        if not amount.is_finite() or amount < 0:
            raise ValueError("minimum tradable amount must be finite and non-negative")
        if self.minimum_listing_days < 0:
            raise ValueError("minimum listing days cannot be negative")
        if not self.allowed_exchanges:
            raise ValueError("tradability market scope must not be empty")
        object.__setattr__(self, "minimum_amount", amount)


@dataclass(frozen=True, slots=True)
class TradabilityDecision:
    allowed: bool
    reasons: tuple[TradabilityReason, ...]


def evaluate_tradability(
    *,
    side: Side,
    bar: DailyBar | None,
    status: SecurityStatus | None,
    rules: TradabilityRules,
) -> TradabilityDecision:
    reasons: list[TradabilityReason] = []
    if bar is None:
        reasons.append(TradabilityReason.MISSING_MARKET_DATA)
    if status is None:
        reasons.append(TradabilityReason.MISSING_STATUS)
    if bar is None or status is None:
        return TradabilityDecision(False, tuple(reasons))
    if bar.symbol != status.symbol or bar.trade_date != status.trade_date:
        raise ValueError("tradability market data and status must align")
    if not status.is_listed:
        reasons.append(TradabilityReason.NOT_LISTED)
    if status.is_delisted:
        reasons.append(TradabilityReason.DELISTED)
    if status.suspended:
        reasons.append(TradabilityReason.SUSPENDED)
    if rules.exclude_st and status.is_st:
        reasons.append(TradabilityReason.ST)
    if side is Side.BUY and status.limit_status is LimitStatus.LIMIT_UP:
        reasons.append(TradabilityReason.LIMIT_UP_BUY)
    if side is Side.SELL and status.limit_status is LimitStatus.LIMIT_DOWN:
        reasons.append(TradabilityReason.LIMIT_DOWN_SELL)
    if bar.high == bar.low and status.limit_status in {
        LimitStatus.LIMIT_UP,
        LimitStatus.LIMIT_DOWN,
    }:
        directional_block = (side is Side.BUY and status.limit_status is LimitStatus.LIMIT_UP) or (
            side is Side.SELL and status.limit_status is LimitStatus.LIMIT_DOWN
        )
        if directional_block:
            reasons.append(TradabilityReason.ONE_PRICE_BOARD)
    if bar.amount < rules.minimum_amount:
        reasons.append(TradabilityReason.INSUFFICIENT_AMOUNT)
    if rules.minimum_listing_days and (
        status.listing_age_days is None or status.listing_age_days < rules.minimum_listing_days
    ):
        reasons.append(TradabilityReason.INSUFFICIENT_LISTING_AGE)
    if bar.symbol.exchange not in rules.allowed_exchanges:
        reasons.append(TradabilityReason.OUTSIDE_MARKET_SCOPE)
    return TradabilityDecision(not reasons, tuple(reasons))


def tradability_mask(
    states: tuple[tuple[DailyBar | None, SecurityStatus | None], ...],
    *,
    side: Side,
    rules: TradabilityRules,
) -> tuple[bool, ...]:
    return tuple(
        evaluate_tradability(side=side, bar=bar, status=status, rules=rules).allowed
        for bar, status in states
    )
