from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from aquant.domain.identifiers import Symbol


@dataclass(frozen=True, slots=True)
class WindMicrocapConstituentReturn:
    trade_date: date
    symbol: Symbol
    simple_return: Decimal

    def __post_init__(self) -> None:
        value = Decimal(self.simple_return)
        if not value.is_finite() or value <= Decimal("-1"):
            raise ValueError("constituent return must be finite and greater than -100%")
        object.__setattr__(self, "simple_return", value)


@dataclass(frozen=True, slots=True)
class WindMicrocapDailyPoint:
    trade_date: date
    simple_return: Decimal
    net_value: Decimal
    one_way_turnover: Decimal
    two_way_turnover: Decimal
    constituent_count: int


def calculate_wind_microcap_daily_equal(
    observations: tuple[WindMicrocapConstituentReturn, ...],
    *,
    target_count: int = 400,
) -> tuple[WindMicrocapDailyPoint, ...]:
    """Calculate a daily-reset arithmetic equal-weight price index.

    Each date's observations are the constituents known at the preceding close.
    Initial deployment is excluded from turnover; subsequent turnover compares
    the new equal weights with the prior basket's post-return drifted weights.
    """

    if target_count <= 0:
        raise ValueError("target count must be positive")
    grouped: dict[date, dict[Symbol, Decimal]] = {}
    for observation in observations:
        values = grouped.setdefault(observation.trade_date, {})
        if observation.symbol in values:
            raise ValueError(
                f"duplicate Wind microcap constituent return: "
                f"{observation.trade_date} {observation.symbol}"
            )
        values[observation.symbol] = observation.simple_return
    if not grouped:
        raise ValueError("Wind microcap return observations must not be empty")

    equal_weight = Decimal("1") / Decimal(target_count)
    prior_post_return_weights: dict[Symbol, Decimal] | None = None
    net_value = Decimal("1")
    output: list[WindMicrocapDailyPoint] = []
    for trade_date, returns in sorted(grouped.items()):
        if len(returns) != target_count:
            raise ValueError(
                f"Wind microcap constituent count is incomplete on {trade_date}: "
                f"{len(returns)} != {target_count}"
            )
        target_weights = {symbol: equal_weight for symbol in returns}
        if prior_post_return_weights is None:
            two_way_turnover = Decimal("0")
        else:
            two_way_turnover = sum(
                (
                    abs(
                        target_weights.get(symbol, Decimal("0"))
                        - prior_post_return_weights.get(symbol, Decimal("0"))
                    )
                    for symbol in set(target_weights) | set(prior_post_return_weights)
                ),
                Decimal("0"),
            )
        simple_return = sum(returns.values(), Decimal("0")) / Decimal(target_count)
        net_value *= Decimal("1") + simple_return
        denominator = Decimal("1") + simple_return
        prior_post_return_weights = {
            symbol: equal_weight * (Decimal("1") + value) / denominator
            for symbol, value in returns.items()
        }
        output.append(
            WindMicrocapDailyPoint(
                trade_date=trade_date,
                simple_return=simple_return,
                net_value=net_value,
                one_way_turnover=two_way_turnover / Decimal("2"),
                two_way_turnover=two_way_turnover,
                constituent_count=target_count,
            )
        )
    return tuple(output)
