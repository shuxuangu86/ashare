from __future__ import annotations

from collections.abc import Mapping, Sequence

from aquant.regime.aggregation import median, weighted_mean
from aquant.regime.definitions import (
    MarketStateFamily,
    MarketStateRole,
    MarketStateScope,
    MarketStateSpec,
)
from aquant.regime.percentiles import expanding_percentile, rolling_percentile
from aquant.regime.protocol import Numeric, StateSeries


def valuation_state_specs() -> tuple[MarketStateSpec, ...]:
    common = {
        "family": MarketStateFamily.VALUATION,
        "required_datasets": ("daily_basic_pit", "index_membership_pit"),
        "availability_lag": 1,
        "role": MarketStateRole.VALUATION_STATE,
    }
    specs = [
        MarketStateSpec(
            state_id="microcap_pb_median",
            display_name="Microcap PB Median",
            subfamily="microcap_pb",
            scope=MarketStateScope.MICROCAP,
            formula="median(pb where pb > 0)",
            parameters={"negative_book_value": "EXCLUDE"},
            required_universes=("MICROCAP_FIXED_MONTHLY",),
            lookback=1,
            minimum_periods=1,
            aggregation_method="CROSS_SECTIONAL_MEDIAN",
            expected_interpretation=(
                "Higher values indicate more expensive microcap book valuation."
            ),
            economic_rationale="The median is robust to extreme PB ratios.",
            **common,
        )
    ]
    for suffix, window, minimum in (("3y", 756, 252), ("5y", 1260, 252)):
        specs.append(
            MarketStateSpec(
                state_id=f"microcap_pb_percentile_{suffix}",
                display_name=f"Microcap PB Rolling {suffix.upper()} Percentile",
                subfamily="microcap_pb",
                scope=MarketStateScope.MICROCAP,
                formula=f"rolling_percentile(microcap_pb_median, {window})",
                parameters={"window": window},
                required_universes=("MICROCAP_FIXED_MONTHLY",),
                lookback=window,
                minimum_periods=minimum,
                aggregation_method="ROLLING_EMPIRICAL_CDF",
                percentile_method=f"ROLLING_{window}D_AVERAGE_TIE",
                expected_interpretation="Values near zero indicate historically cheap microcaps.",
                economic_rationale="Rolling history prevents full-sample valuation leakage.",
                **common,
            )
        )
    specs.extend(
        (
            MarketStateSpec(
                state_id="microcap_pb_percentile_expanding",
                display_name="Microcap PB Expanding Percentile",
                subfamily="microcap_pb",
                scope=MarketStateScope.MICROCAP,
                formula="expanding_percentile(microcap_pb_median)",
                parameters={"minimum_periods": 252},
                required_universes=("MICROCAP_FIXED_MONTHLY",),
                lookback=252,
                minimum_periods=252,
                aggregation_method="EXPANDING_EMPIRICAL_CDF",
                percentile_method="EXPANDING_PAST_ONLY_AVERAGE_TIE",
                expected_interpretation=(
                    "Values near zero indicate cheapness versus all prior history."
                ),
                economic_rationale="Expanding percentiles use no future observations.",
                **common,
            ),
            MarketStateSpec(
                state_id="dividend_index_dividend_yield_mean",
                display_name="Dividend Index Dividend Yield Mean",
                subfamily="dividend_yield",
                scope=MarketStateScope.DIVIDEND,
                formula="mean(dividend_yield where dividend_yield >= 0)",
                parameters={},
                required_indices=("DIVIDEND_INDEX",),
                lookback=1,
                minimum_periods=1,
                aggregation_method="CROSS_SECTIONAL_MEAN",
                expected_interpretation="Higher values indicate higher constituent cash yield.",
                economic_rationale="The simple mean describes the average index constituent.",
                **common,
            ),
            MarketStateSpec(
                state_id="dividend_index_dividend_yield_median",
                display_name="Dividend Index Dividend Yield Median",
                subfamily="dividend_yield",
                scope=MarketStateScope.DIVIDEND,
                formula="median(dividend_yield where dividend_yield >= 0)",
                parameters={},
                required_indices=("DIVIDEND_INDEX",),
                lookback=1,
                minimum_periods=1,
                aggregation_method="CROSS_SECTIONAL_MEDIAN",
                expected_interpretation=(
                    "Higher values indicate broad-based constituent cash yield."
                ),
                economic_rationale="The median reduces sensitivity to a few extreme yields.",
                **common,
            ),
            MarketStateSpec(
                state_id="dividend_index_dividend_yield_ffmv_weighted",
                display_name="Dividend Index FFMV-weighted Dividend Yield",
                subfamily="dividend_yield",
                scope=MarketStateScope.DIVIDEND,
                formula="sum(dividend_yield * free_float_mv) / sum(free_float_mv)",
                parameters={},
                required_indices=("DIVIDEND_INDEX",),
                lookback=1,
                minimum_periods=1,
                aggregation_method="FREE_FLOAT_MARKET_CAP_WEIGHTED_MEAN",
                expected_interpretation=(
                    "Higher values indicate higher investable index cash yield."
                ),
                economic_rationale="Free-float weights approximate an investable portfolio.",
                **common,
            ),
            MarketStateSpec(
                state_id="dividend_yield_percentile_3y",
                display_name="Dividend Yield Rolling 3Y Percentile",
                subfamily="dividend_yield",
                scope=MarketStateScope.DIVIDEND,
                formula="rolling_percentile(dividend_index_dividend_yield_ffmv_weighted, 756)",
                parameters={"window": 756},
                required_indices=("DIVIDEND_INDEX",),
                lookback=756,
                minimum_periods=252,
                aggregation_method="ROLLING_EMPIRICAL_CDF",
                percentile_method="ROLLING_756D_AVERAGE_TIE",
                expected_interpretation="Values near one indicate unusually high dividend yield.",
                economic_rationale="Past-only yield percentile contextualizes income valuation.",
                **common,
            ),
        )
    )
    return tuple(specs)


def compute_microcap_pb_states(pb_rows: Sequence[Sequence[Numeric]]) -> Mapping[str, StateSeries]:
    medians = tuple(median(row, positive_only=True) for row in pb_rows)
    return {
        "microcap_pb_median": medians,
        "microcap_pb_percentile_3y": rolling_percentile(medians, 756, minimum_periods=252),
        "microcap_pb_percentile_5y": rolling_percentile(medians, 1260, minimum_periods=252),
        "microcap_pb_percentile_expanding": expanding_percentile(medians, minimum_periods=252),
    }


def compute_dividend_yield_states(
    yield_rows: Sequence[Sequence[Numeric]],
    free_float_market_value_rows: Sequence[Sequence[Numeric]],
) -> Mapping[str, StateSeries]:
    if len(yield_rows) != len(free_float_market_value_rows):
        raise ValueError("dividend-yield rows and weight rows must have equal length")
    means: list[float | None] = []
    medians: list[float | None] = []
    weighted: list[float | None] = []
    for yields, weights in zip(yield_rows, free_float_market_value_rows, strict=True):
        valid_yields = [float(value) for value in yields if value is not None and float(value) >= 0]
        means.append(sum(valid_yields) / len(valid_yields) if valid_yields else None)
        medians.append(median(valid_yields))
        weighted.append(
            weighted_mean(
                [value if value is not None and float(value) >= 0 else None for value in yields],
                weights,
            )
        )
    weighted_series = tuple(weighted)
    return {
        "dividend_index_dividend_yield_mean": tuple(means),
        "dividend_index_dividend_yield_median": tuple(medians),
        "dividend_index_dividend_yield_ffmv_weighted": weighted_series,
        "dividend_yield_percentile_3y": rolling_percentile(
            weighted_series, 756, minimum_periods=252
        ),
    }
