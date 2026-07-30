from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from aquant.regime.definitions import (
    MarketStateFamily,
    MarketStateReleaseStatus,
    MarketStateRole,
    MarketStateScope,
    MarketStateSpec,
)
from aquant.regime.protocol import Numeric, StateSeries

CROWDING_INPUTS = (
    "microcap_amount_share",
    "microcap_turnover",
    "microcap_relative_valuation",
    "microcap_relative_momentum",
    "microcap_breadth",
)


def crowding_state_specs() -> tuple[MarketStateSpec, ...]:
    specs: list[MarketStateSpec] = []
    for state_id in CROWDING_INPUTS:
        specs.append(
            MarketStateSpec(
                state_id=state_id,
                display_name=state_id.replace("_", " ").title(),
                family=MarketStateFamily.CROWDING,
                subfamily="microcap_crowding_input",
                scope=MarketStateScope.MICROCAP,
                formula=f"past_only_percentile({state_id}_raw)",
                parameters={"direction": "HIGH_IS_CROWDED"},
                required_datasets=("bars_1d", "daily_basic_pit", "universe_membership_pit"),
                required_universes=("MICROCAP_FIXED_MONTHLY", "ALL_A"),
                lookback=756,
                minimum_periods=252,
                aggregation_method="ROLLING_EMPIRICAL_CDF",
                normalization="PERCENTILE_0_1",
                percentile_method="ROLLING_756D_AVERAGE_TIE",
                expected_interpretation="Higher values indicate a more crowded microcap input.",
                economic_rationale=(
                    "Separate activity, valuation, momentum, and breadth inputs are retained."
                ),
                role=MarketStateRole.CROWDING_STATE,
                correlation_cluster="microcap_crowding_inputs",
                related_states=tuple(item for item in CROWDING_INPUTS if item != state_id),
            )
        )
    specs.append(
        MarketStateSpec(
            state_id="microcap_crowding_score_v1",
            display_name="Microcap Crowding Score v1",
            family=MarketStateFamily.CROWDING,
            subfamily="microcap_crowding_composite",
            scope=MarketStateScope.MICROCAP,
            formula="mean(available registered microcap crowding percentile inputs)",
            parameters={"minimum_components": 3, "equal_weighted": True},
            required_datasets=("market_state_values",),
            required_universes=("MICROCAP_FIXED_MONTHLY",),
            lookback=1,
            minimum_periods=1,
            aggregation_method="AVAILABLE_COMPONENT_MEAN",
            normalization="PERCENTILE_0_1",
            percentile_method="COMPONENT_PAST_ONLY_PERCENTILES",
            expected_interpretation="Higher values indicate broader microcap crowding pressure.",
            economic_rationale="A smoke composite supports research while preserving every input.",
            role=MarketStateRole.CROWDING_STATE,
            release_status=MarketStateReleaseStatus.DRAFT,
            related_states=CROWDING_INPUTS,
        )
    )
    return tuple(specs)


def microcap_crowding_score(
    inputs: Mapping[str, Sequence[Numeric]],
    *,
    minimum_components: int = 3,
) -> StateSeries:
    missing = set(CROWDING_INPUTS).difference(inputs)
    if missing:
        raise ValueError(f"missing microcap crowding inputs: {sorted(missing)}")
    lengths = {len(inputs[name]) for name in CROWDING_INPUTS}
    if len(lengths) != 1:
        raise ValueError("microcap crowding inputs must have equal length")
    if not 1 <= minimum_components <= len(CROWDING_INPUTS):
        raise ValueError("minimum_components is outside the available input range")
    output: list[float | None] = []
    for index in range(lengths.pop()):
        values: list[float] = []
        for name in CROWDING_INPUTS:
            value = inputs[name][index]
            if value is None:
                continue
            resolved = float(value)
            if math.isfinite(resolved) and 0.0 <= resolved <= 1.0:
                values.append(resolved)
        output.append(sum(values) / len(values) if len(values) >= minimum_components else None)
    return tuple(output)
