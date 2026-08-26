from __future__ import annotations

from collections.abc import Mapping, Sequence

from aquant.regime.aggregation import amount_share, rolling_mean
from aquant.regime.definitions import (
    MarketStateFamily,
    MarketStateRole,
    MarketStateScope,
    MarketStateSpec,
)
from aquant.regime.percentiles import rolling_percentile
from aquant.regime.protocol import Numeric, StateSeries


def liquidity_state_specs() -> tuple[MarketStateSpec, ...]:
    common = {
        "family": MarketStateFamily.LIQUIDITY,
        "required_datasets": ("bars_1d", "index_membership_pit"),
        "availability_lag": 1,
        "role": MarketStateRole.LIQUIDITY_STATE,
    }
    return (
        MarketStateSpec(
            state_id="hs300_amount_share_all_a",
            display_name="HS300 Amount Share of All A",
            subfamily="amount_share",
            scope=MarketStateScope.HS300,
            formula="sum(hs300_member_amount) / sum(all_a_tradable_amount)",
            parameters={},
            required_indices=("HS300",),
            required_universes=("ALL_A",),
            lookback=1,
            minimum_periods=1,
            aggregation_method="RATIO_OF_SUMS",
            expected_interpretation=(
                "Higher values mean trading activity concentrates in large caps."
            ),
            economic_rationale=(
                "Amount is comparable across securities and proxies capital attention."
            ),
            **common,
        ),
        MarketStateSpec(
            state_id="hs300_amount_share_20d",
            display_name="HS300 Amount Share 20D Mean",
            subfamily="amount_share",
            scope=MarketStateScope.HS300,
            formula="mean(hs300_amount_share_all_a, 20)",
            parameters={"window": 20},
            required_indices=("HS300",),
            required_universes=("ALL_A",),
            lookback=20,
            minimum_periods=20,
            aggregation_method="ROLLING_MEAN",
            expected_interpretation="Higher values mean sustained large-cap trading concentration.",
            economic_rationale="Smoothing removes one-session amount noise.",
            **common,
        ),
        MarketStateSpec(
            state_id="hs300_amount_share_percentile_3y",
            display_name="HS300 Amount Share Rolling 3Y Percentile",
            subfamily="amount_share",
            scope=MarketStateScope.HS300,
            formula="rolling_percentile(hs300_amount_share_all_a, 756)",
            parameters={"window": 756},
            required_indices=("HS300",),
            required_universes=("ALL_A",),
            lookback=756,
            minimum_periods=252,
            aggregation_method="ROLLING_EMPIRICAL_CDF",
            percentile_method="ROLLING_756D_AVERAGE_TIE",
            expected_interpretation="Values near one indicate unusually high large-cap attention.",
            economic_rationale="Past-only percentiles make activity comparable across regimes.",
            **common,
        ),
        MarketStateSpec(
            state_id="csi1000_turnover_daily",
            display_name="CSI1000 Aggregate Turnover Daily",
            subfamily="turnover",
            scope=MarketStateScope.CSI1000,
            formula="sum(member_amount) / sum(member_free_float_market_value)",
            parameters={},
            required_indices=("CSI1000",),
            required_universes=(),
            lookback=1,
            minimum_periods=1,
            aggregation_method="PORTFOLIO_AGGREGATE_TURNOVER",
            expected_interpretation="Higher values indicate more active small-cap trading.",
            economic_rationale="Ratio-of-sums is an explicit, capacity-relevant turnover measure.",
            **common,
        ),
        MarketStateSpec(
            state_id="csi1000_turnover_20d",
            display_name="CSI1000 Aggregate Turnover 20D",
            subfamily="turnover",
            scope=MarketStateScope.CSI1000,
            formula="mean(csi1000_turnover_daily, 20)",
            parameters={"window": 20},
            required_indices=("CSI1000",),
            required_universes=(),
            lookback=20,
            minimum_periods=20,
            aggregation_method="ROLLING_MEAN",
            expected_interpretation="Higher values indicate persistent small-cap trading activity.",
            economic_rationale="Twenty-session turnover captures sustained participation.",
            **common,
        ),
        MarketStateSpec(
            state_id="csi1000_turnover_percentile_3y",
            display_name="CSI1000 Turnover Rolling 3Y Percentile",
            subfamily="turnover",
            scope=MarketStateScope.CSI1000,
            formula="rolling_percentile(csi1000_turnover_daily, 756)",
            parameters={"window": 756},
            required_indices=("CSI1000",),
            required_universes=(),
            lookback=756,
            minimum_periods=252,
            aggregation_method="ROLLING_EMPIRICAL_CDF",
            percentile_method="ROLLING_756D_AVERAGE_TIE",
            expected_interpretation="Values near one indicate historically elevated turnover.",
            economic_rationale="Past-only normalization avoids future leakage.",
            **common,
        ),
    )


def compute_liquidity_states(
    hs300_amount: Sequence[Numeric],
    all_a_amount: Sequence[Numeric],
    csi1000_turnover: Sequence[Numeric],
) -> Mapping[str, StateSeries]:
    share = amount_share(hs300_amount, all_a_amount)
    return {
        "hs300_amount_share_all_a": share,
        "hs300_amount_share_20d": rolling_mean(share, 20),
        "hs300_amount_share_percentile_3y": rolling_percentile(share, 756, minimum_periods=252),
        "csi1000_turnover_daily": tuple(
            None if value is None else float(value) for value in csi1000_turnover
        ),
        "csi1000_turnover_20d": rolling_mean(csi1000_turnover, 20),
        "csi1000_turnover_percentile_3y": rolling_percentile(
            csi1000_turnover, 756, minimum_periods=252
        ),
    }
