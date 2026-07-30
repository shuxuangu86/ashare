from __future__ import annotations

from collections.abc import Mapping, Sequence

from aquant.regime.aggregation import relative_returns
from aquant.regime.definitions import (
    MarketStateFamily,
    MarketStateRole,
    MarketStateScope,
    MarketStateSpec,
)
from aquant.regime.protocol import Numeric, StateSeries

INDEX_RELATIVE_WINDOWS = (5, 20, 40, 60, 120, 252)


def index_relative_state_specs() -> tuple[MarketStateSpec, ...]:
    return tuple(
        MarketStateSpec(
            state_id=f"hs300_minus_csi2000_return_{window}d",
            display_name=f"HS300 minus CSI2000 {window}D Return",
            family=MarketStateFamily.INDEX_RELATIVE,
            subfamily="large_minus_small",
            scope=MarketStateScope.ALL_A,
            formula=(
                f"(hs300_close / delay(hs300_close, {window})) / "
                f"(csi2000_close / delay(csi2000_close, {window})) - 1"
            ),
            parameters={"window": window, "left": "HS300", "right": "CSI2000"},
            required_datasets=("index_bars_1d",),
            required_indices=("HS300", "CSI2000"),
            lookback=window,
            minimum_periods=window,
            aggregation_method="COMPOUNDED_RELATIVE_RETURN",
            expected_interpretation="Positive values indicate large-cap relative strength.",
            economic_rationale=(
                "HS300 versus CSI2000 captures the large-small capitalization cycle."
            ),
            role=MarketStateRole.REGIME_INPUT,
            correlation_cluster="hs300_minus_csi2000_return",
            related_states=tuple(
                f"hs300_minus_csi2000_return_{other}d"
                for other in INDEX_RELATIVE_WINDOWS
                if other != window
            ),
        )
        for window in INDEX_RELATIVE_WINDOWS
    )


def compute_hs300_csi2000_states(
    hs300_close: Sequence[Numeric],
    csi2000_close: Sequence[Numeric],
) -> Mapping[str, StateSeries]:
    return {
        f"hs300_minus_csi2000_return_{window}d": relative_returns(
            hs300_close, csi2000_close, window
        )
        for window in INDEX_RELATIVE_WINDOWS
    }
