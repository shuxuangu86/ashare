from datetime import date, datetime
from decimal import Decimal

from aquant.domain.portfolio import TargetPortfolio, TargetPosition
from aquant.portfolio.signals.multifactor import ScoredSecurity


def construct_top_n_equal_weight(
    scores: tuple[ScoredSecurity, ...],
    *,
    top_n: int,
    maximum_weight: Decimal,
    minimum_cash_weight: Decimal,
    strategy_id: str,
    trade_date: date,
    asof_time: datetime,
    data_release_id: str,
    signal_version: str,
) -> TargetPortfolio:
    if top_n <= 0 or not scores:
        raise ValueError("Top-N construction requires candidates and positive top_n")
    selected = scores[:top_n]
    investable = Decimal("1") - Decimal(minimum_cash_weight)
    if not Decimal("0") <= investable <= Decimal("1") or maximum_weight <= 0:
        raise ValueError("portfolio cash/weight constraints are invalid")
    equal_weight = min(investable / len(selected), Decimal(maximum_weight))
    positions = tuple(TargetPosition(item.symbol, equal_weight) for item in selected)
    return TargetPortfolio(
        strategy_id,
        trade_date,
        asof_time,
        data_release_id,
        signal_version,
        positions,
    )
