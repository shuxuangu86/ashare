from collections.abc import Callable

import numpy as np

from aquant.factors.atomic.models import Array, AtomicFactor, FactorPanelInput, atomic_factor
from aquant.factors.operators.math import log, safe_div
from aquant.factors.operators.time_series import (
    count_if,
    days_since_high,
    delta,
    regression_slope,
    returns,
    rolling_corr,
    rolling_max,
    rolling_mean,
    rolling_std,
    rolling_sum,
)


def _field(panel: FactorPanelInput, name: str) -> Array:
    return np.asarray(panel.fields[name], dtype=np.float64)


def point_factor(
    factor_id: str,
    family: str,
    field: str,
    transform: Callable[[Array], Array],
    description: str,
    hypothesis: str,
    *,
    direction: int = 0,
) -> AtomicFactor:
    return atomic_factor(
        factor_id,
        family,
        description,
        hypothesis,
        (field,),
        1,
        lambda panel: transform(_field(panel, field)),
        expected_direction=direction,
        parameters={"field": field},
    )


def ratio_factor(
    factor_id: str,
    family: str,
    numerator: str,
    denominator: str,
    description: str,
    hypothesis: str,
    *,
    direction: int = 0,
) -> AtomicFactor:
    return atomic_factor(
        factor_id,
        family,
        description,
        hypothesis,
        (numerator, denominator),
        1,
        lambda panel: safe_div(_field(panel, numerator), _field(panel, denominator)),
        expected_direction=direction,
        parameters={"numerator": numerator, "denominator": denominator},
    )


def momentum_factor(window: int) -> AtomicFactor:
    return atomic_factor(
        f"momentum_{window}d",
        "momentum",
        f"Close-to-close return over {window} sessions.",
        "Medium-horizon price strength may persist.",
        ("close",),
        window + 1,
        lambda panel: returns(_field(panel, "close"), window),
        expected_direction=1,
        parameters={"window": window},
    )


def reversal_factor(window: int) -> AtomicFactor:
    return atomic_factor(
        f"reversal_{window}d",
        "reversal",
        f"Negative close return over {window} sessions.",
        "Short-horizon price shocks may mean-revert.",
        ("close",),
        window + 1,
        lambda panel: -returns(_field(panel, "close"), window),
        expected_direction=1,
        parameters={"window": window},
    )


def volatility_factor(window: int) -> AtomicFactor:
    return atomic_factor(
        f"volatility_{window}d",
        "volatility",
        f"Trailing {window}-session standard deviation of daily returns.",
        "Lower realized volatility may be rewarded in long-only portfolios.",
        ("close",),
        window + 1,
        lambda panel: rolling_std(returns(_field(panel, "close")), window),
        expected_direction=-1,
        parameters={"window": window, "ddof": 1},
    )


def downside_volatility_factor(window: int) -> AtomicFactor:
    def calculate(panel: FactorPanelInput) -> Array:
        daily_returns = returns(_field(panel, "close"))
        return rolling_std(np.where(daily_returns < 0, daily_returns, 0), window)

    return atomic_factor(
        f"downside_volatility_{window}d",
        "volatility",
        f"Trailing {window}-session downside-return volatility.",
        "Concentrated downside variation indicates adverse risk.",
        ("close",),
        window + 1,
        calculate,
        expected_direction=-1,
        parameters={"window": window},
    )


def rolling_mean_factor(
    factor_id: str,
    family: str,
    field: str,
    window: int,
    description: str,
    hypothesis: str,
    *,
    direction: int = 0,
) -> AtomicFactor:
    return atomic_factor(
        factor_id,
        family,
        description,
        hypothesis,
        (field,),
        window,
        lambda panel: rolling_mean(_field(panel, field), window),
        expected_direction=direction,
        parameters={"field": field, "window": window},
    )


def rolling_change_factor(
    factor_id: str,
    family: str,
    field: str,
    window: int,
    description: str,
    hypothesis: str,
    *,
    direction: int = 0,
) -> AtomicFactor:
    return atomic_factor(
        factor_id,
        family,
        description,
        hypothesis,
        (field,),
        window + 1,
        lambda panel: delta(_field(panel, field), window),
        expected_direction=direction,
        parameters={"field": field, "window": window},
    )


def rolling_correlation_factor(
    factor_id: str,
    family: str,
    left: str,
    right: str,
    window: int,
    description: str,
    hypothesis: str,
) -> AtomicFactor:
    return atomic_factor(
        factor_id,
        family,
        description,
        hypothesis,
        (left, right),
        window,
        lambda panel: rolling_corr(_field(panel, left), _field(panel, right), window),
        parameters={"left": left, "right": right, "window": window},
    )


def trend_slope_factor(window: int) -> AtomicFactor:
    return atomic_factor(
        f"trend_slope_{window}d",
        "momentum",
        f"Trailing {window}-session slope of log close.",
        "Persistent price trends have a positive log-price slope.",
        ("close",),
        window,
        lambda panel: regression_slope(log(_field(panel, "close")), window),
        expected_direction=1,
        parameters={"window": window},
    )


def distance_to_high_factor(window: int) -> AtomicFactor:
    return atomic_factor(
        f"distance_to_{window}d_high",
        "momentum",
        f"Close divided by its trailing {window}-session high minus one.",
        "Stocks near established highs may retain price strength.",
        ("close",),
        window,
        lambda panel: (
            safe_div(
                _field(panel, "close"),
                rolling_max(_field(panel, "close"), window),
            )
            - 1
        ),
        expected_direction=1,
        parameters={"window": window},
    )


def zero_return_days_factor(window: int) -> AtomicFactor:
    return atomic_factor(
        f"zero_return_days_{window}d",
        "liquidity",
        f"Count of zero-return sessions over {window} sessions.",
        "Frequent zero returns proxy weak trading activity.",
        ("close",),
        window + 1,
        lambda panel: count_if(np.isclose(returns(_field(panel, "close")), 0), window),
        expected_direction=-1,
        parameters={"window": window},
    )


def amihud_factor(window: int) -> AtomicFactor:
    return atomic_factor(
        f"amihud_{window}d",
        "liquidity",
        f"Mean absolute return per traded amount over {window} sessions.",
        "Large price moves per currency traded indicate illiquidity.",
        ("close", "amount"),
        window + 1,
        lambda panel: rolling_mean(
            safe_div(np.abs(returns(_field(panel, "close"))), _field(panel, "amount")),
            window,
        ),
        expected_direction=-1,
        parameters={"window": window},
    )


def limit_frequency_factor(window: int) -> AtomicFactor:
    return atomic_factor(
        f"limit_up_frequency_{window}d",
        "price_volume",
        f"Frequency of closes at the upper limit over {window} sessions.",
        "Repeated limit closes capture crowding and execution constraints.",
        ("close", "up_limit"),
        window,
        lambda panel: safe_div(
            rolling_sum(
                np.isclose(_field(panel, "close"), _field(panel, "up_limit")).astype(float),
                window,
            ),
            window,
        ),
        expected_direction=0,
        parameters={"window": window},
    )


def days_from_high_factor(window: int) -> AtomicFactor:
    return atomic_factor(
        f"days_since_{window}d_high",
        "price_volume",
        f"Sessions elapsed since the trailing {window}-session close high.",
        "Recent highs distinguish persistent trends from stale breakouts.",
        ("close",),
        window,
        lambda panel: days_since_high(_field(panel, "close"), window),
        parameters={"window": window},
    )
