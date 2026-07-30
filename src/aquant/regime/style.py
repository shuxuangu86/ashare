from __future__ import annotations

from collections.abc import Mapping, Sequence

from aquant.regime.aggregation import relative_returns, returns
from aquant.regime.definitions import (
    MarketStateFamily,
    MarketStateRole,
    MarketStateScope,
    MarketStateSpec,
)
from aquant.regime.protocol import Numeric, StateSeries

STYLE_WINDOWS = (5, 20, 40, 60, 120)


def style_state_specs() -> tuple[MarketStateSpec, ...]:
    specs: list[MarketStateSpec] = []
    for style, scope in (
        ("growth", MarketStateScope.GROWTH),
        ("value", MarketStateScope.VALUE),
    ):
        for window in STYLE_WINDOWS:
            specs.append(
                MarketStateSpec(
                    state_id=f"{style}_return_{window}d",
                    display_name=f"{style.title()} {window}D Return",
                    family=MarketStateFamily.STYLE,
                    subfamily="style_absolute_return",
                    scope=scope,
                    formula=f"close / delay(close, {window}) - 1",
                    parameters={"window": window, "style": style},
                    required_datasets=("index_bars_1d",),
                    required_indices=(style.upper(),),
                    lookback=window,
                    minimum_periods=window,
                    aggregation_method="INDEX_CLOSE_RETURN",
                    expected_interpretation=f"Higher means stronger {style} performance.",
                    economic_rationale="Style-index return measures the prevailing style tape.",
                    role=MarketStateRole.STYLE_STATE,
                    correlation_cluster=f"{style}_return",
                )
            )
    for window in STYLE_WINDOWS:
        specs.append(
            MarketStateSpec(
                state_id=f"growth_minus_value_return_{window}d",
                display_name=f"Growth minus Value {window}D Return",
                family=MarketStateFamily.STYLE,
                subfamily="growth_value_dominance",
                scope=MarketStateScope.ALL_A,
                formula=(
                    f"(growth_close / delay(growth_close, {window})) / "
                    f"(value_close / delay(value_close, {window})) - 1"
                ),
                parameters={"window": window},
                required_datasets=("index_bars_1d",),
                required_indices=("GROWTH", "VALUE"),
                lookback=window,
                minimum_periods=window,
                aggregation_method="COMPOUNDED_RELATIVE_RETURN",
                expected_interpretation="Positive values indicate growth dominance over value.",
                economic_rationale=(
                    "Relative compounded return isolates the growth-value style cycle."
                ),
                role=MarketStateRole.REGIME_INPUT,
                correlation_cluster="growth_minus_value_return",
                related_states=tuple(
                    f"growth_minus_value_return_{other}d"
                    for other in STYLE_WINDOWS
                    if other != window
                ),
            )
        )
    return tuple(specs)


def compute_style_states(
    growth_close: Sequence[Numeric],
    value_close: Sequence[Numeric],
) -> Mapping[str, StateSeries]:
    if len(growth_close) != len(value_close):
        raise ValueError("growth and value series must have equal length")
    output: dict[str, StateSeries] = {}
    for window in STYLE_WINDOWS:
        output[f"growth_return_{window}d"] = returns(growth_close, window)
        output[f"value_return_{window}d"] = returns(value_close, window)
        output[f"growth_minus_value_return_{window}d"] = relative_returns(
            growth_close, value_close, window
        )
    return output
