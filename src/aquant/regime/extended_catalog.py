from __future__ import annotations

from collections.abc import Iterable

from aquant.regime.definitions import (
    MarketStateFamily,
    MarketStateRole,
    MarketStateScope,
    MarketStateSpec,
    MarketStateStatus,
)

_MARKET_SCOPES = (
    ("all_a", MarketStateScope.ALL_A, "ALL_A"),
    ("hs300", MarketStateScope.HS300, "HS300"),
    ("csi500", MarketStateScope.CSI500, "CSI500"),
    ("csi1000", MarketStateScope.CSI1000, "CSI1000"),
    ("microcap", MarketStateScope.MICROCAP, "MICROCAP_FIXED_MONTHLY"),
    ("chinext", MarketStateScope.GROWTH, "CHINEXT"),
)

_BREADTH_SCOPES = (
    *_MARKET_SCOPES[:5],
    ("growth", MarketStateScope.GROWTH, "GROWTH_DYNAMIC"),
    ("value", MarketStateScope.VALUE, "VALUE_DYNAMIC"),
    ("dividend", MarketStateScope.DIVIDEND, "DIVIDEND_DYNAMIC"),
)


def _spec(
    *,
    state_id: str,
    family: MarketStateFamily,
    subfamily: str,
    scope: MarketStateScope,
    formula: str,
    lookback: int,
    aggregation: str,
    interpretation: str,
    universe: str,
    role: MarketStateRole = MarketStateRole.STATE_FEATURE,
    status: MarketStateStatus = MarketStateStatus.IMPLEMENTED,
    parameters: dict[str, object] | None = None,
    datasets: tuple[str, ...] = ("daily",),
) -> MarketStateSpec:
    return MarketStateSpec(
        state_id=state_id,
        display_name=state_id.replace("_", " ").title(),
        family=family,
        subfamily=subfamily,
        scope=scope,
        formula=formula,
        parameters=parameters or {},
        required_datasets=datasets,
        required_universes=(universe,),
        lookback=lookback,
        minimum_periods=max(1, min(lookback, 20)),
        aggregation_method=aggregation,
        expected_interpretation=interpretation,
        economic_rationale=(
            "The state retains a continuous, point-in-time description of the market environment."
        ),
        role=role,
        status=status,
        correlation_cluster=f"{subfamily}_{universe.lower()}",
    )


def trend_state_specs() -> tuple[MarketStateSpec, ...]:
    specs: list[MarketStateSpec] = []
    for prefix, scope, universe in _MARKET_SCOPES:
        for window in (5, 20, 40, 60, 120):
            specs.append(
                _spec(
                    state_id=f"{prefix}_return_{window}d",
                    family=MarketStateFamily.TREND,
                    subfamily="market_return",
                    scope=scope,
                    formula=f"close_index / delay(close_index, {window}) - 1",
                    lookback=window,
                    aggregation="COMPOUNDED_RETURN",
                    interpretation="Higher values indicate stronger trailing performance.",
                    universe=universe,
                    role=MarketStateRole.REGIME_INPUT,
                    parameters={"window": window},
                )
            )
        for window in (20, 60, 120):
            specs.extend(
                (
                    _spec(
                        state_id=f"{prefix}_drawdown_{window}d",
                        family=MarketStateFamily.TREND,
                        subfamily="drawdown",
                        scope=scope,
                        formula=f"close_index / rolling_max(close_index, {window}) - 1",
                        lookback=window,
                        aggregation="ROLLING_PEAK_DRAWDOWN",
                        interpretation="More negative values indicate a deeper drawdown.",
                        universe=universe,
                        role=MarketStateRole.RISK_STATE,
                        parameters={"window": window},
                    ),
                    _spec(
                        state_id=f"{prefix}_distance_to_ma{window}",
                        family=MarketStateFamily.TREND,
                        subfamily="distance_to_average",
                        scope=scope,
                        formula=f"close_index / mean(close_index, {window}) - 1",
                        lookback=window,
                        aggregation="DISTANCE_TO_ROLLING_MEAN",
                        interpretation="Positive values indicate price above its moving average.",
                        universe=universe,
                        parameters={"window": window},
                    ),
                    _spec(
                        state_id=f"{prefix}_realized_volatility_{window}d",
                        family=MarketStateFamily.VOLATILITY,
                        subfamily="realized_volatility",
                        scope=scope,
                        formula=f"std(log_return, {window}) * sqrt(252)",
                        lookback=window,
                        aggregation="ANNUALIZED_REALIZED_VOLATILITY",
                        interpretation="Higher values indicate a more volatile market.",
                        universe=universe,
                        role=MarketStateRole.RISK_STATE,
                        parameters={"window": window},
                    ),
                )
            )
        for window in (20, 60):
            specs.append(
                _spec(
                    state_id=f"{prefix}_trend_slope_{window}d",
                    family=MarketStateFamily.TREND,
                    subfamily="trend_slope",
                    scope=scope,
                    formula=f"ols_slope(log(close_index), {window})",
                    lookback=window,
                    aggregation="ROLLING_OLS_SLOPE",
                    interpretation="Positive values indicate an upward fitted trend.",
                    universe=universe,
                    parameters={"window": window},
                )
            )
    return tuple(specs)


def breadth_state_specs() -> tuple[MarketStateSpec, ...]:
    specs: list[MarketStateSpec] = []
    for prefix, scope, universe in _BREADTH_SCOPES:
        status = (
            MarketStateStatus.PARTIAL if universe == "CSI1000" else MarketStateStatus.IMPLEMENTED
        )
        metrics: tuple[tuple[str, str, int, str], ...] = (
            ("positive_return_ratio", "mean(return_1d > 0)", 1, "ADVANCING_SHARE"),
            ("above_ma5_ratio", "mean(close > ma5)", 5, "ABOVE_MOVING_AVERAGE_SHARE"),
            ("above_ma20_ratio", "mean(close > ma20)", 20, "ABOVE_MOVING_AVERAGE_SHARE"),
            ("above_ma60_ratio", "mean(close > ma60)", 60, "ABOVE_MOVING_AVERAGE_SHARE"),
            ("above_ma120_ratio", "mean(close > ma120)", 120, "ABOVE_MOVING_AVERAGE_SHARE"),
            ("new_high_60d_ratio", "mean(close >= rolling_max(close, 60))", 60, "NEW_HIGH_SHARE"),
            (
                "new_high_252d_ratio",
                "mean(close >= rolling_max(close, 252))",
                252,
                "NEW_HIGH_SHARE",
            ),
            ("new_low_60d_ratio", "mean(close <= rolling_min(close, 60))", 60, "NEW_LOW_SHARE"),
            (
                "new_low_252d_ratio",
                "mean(close <= rolling_min(close, 252))",
                252,
                "NEW_LOW_SHARE",
            ),
            (
                "advance_decline_ratio",
                "advancing_count / max(declining_count, 1)",
                1,
                "ADVANCE_DECLINE_RATIO",
            ),
            ("limit_up_ratio", "mean(close >= up_limit)", 1, "LIMIT_UP_SHARE"),
            ("limit_down_ratio", "mean(close <= down_limit)", 1, "LIMIT_DOWN_SHARE"),
            ("large_gain_ratio", "mean(return_1d >= 0.05)", 1, "LARGE_GAIN_SHARE"),
            ("large_loss_ratio", "mean(return_1d <= -0.05)", 1, "LARGE_LOSS_SHARE"),
        )
        for suffix, formula, lookback, aggregation in metrics:
            specs.append(
                _spec(
                    state_id=f"{prefix}_{suffix}",
                    family=MarketStateFamily.BREADTH,
                    subfamily=suffix,
                    scope=scope,
                    formula=formula,
                    lookback=lookback,
                    aggregation=aggregation,
                    interpretation=(
                        "Higher values indicate broader participation in this condition."
                    ),
                    universe=universe,
                    role=MarketStateRole.RISK_STATE,
                    status=status,
                    datasets=("daily", "stk_limit", "index_weight"),
                )
            )
    return tuple(specs)


def valuation_extension_specs() -> tuple[MarketStateSpec, ...]:
    specs: list[MarketStateSpec] = []
    scopes = (
        ("all_a", MarketStateScope.ALL_A, "ALL_A"),
        ("hs300", MarketStateScope.HS300, "HS300"),
        ("csi500", MarketStateScope.CSI500, "CSI500"),
        ("csi1000", MarketStateScope.CSI1000, "CSI1000"),
        ("microcap", MarketStateScope.MICROCAP, "MICROCAP_FIXED_MONTHLY"),
        ("growth", MarketStateScope.GROWTH, "GROWTH_DYNAMIC"),
        ("value", MarketStateScope.VALUE, "VALUE_DYNAMIC"),
        ("dividend", MarketStateScope.DIVIDEND, "DIVIDEND_DYNAMIC"),
    )
    existing = {
        "microcap_pb_median",
        "microcap_pb_percentile_3y",
        "microcap_pb_percentile_5y",
        "microcap_pb_percentile_expanding",
    }
    metrics: tuple[tuple[str, str, int, str], ...] = (
        ("pb_median", "median(pb where pb > 0)", 1, "CROSS_SECTIONAL_MEDIAN"),
        (
            "pb_winsorized_mean",
            "mean(winsorize(pb where pb > 0, 0.01, 0.99))",
            1,
            "WINSORIZED_MEAN",
        ),
        (
            "aggregate_pb",
            "sum(total_mv) / sum(total_mv / pb where pb > 0)",
            1,
            "RATIO_OF_SUMS",
        ),
        ("pb_25pct", "quantile(pb where pb > 0, 0.25)", 1, "CROSS_SECTIONAL_QUANTILE"),
        ("pb_75pct", "quantile(pb where pb > 0, 0.75)", 1, "CROSS_SECTIONAL_QUANTILE"),
        ("pb_dispersion", "std(pb where pb > 0)", 1, "CROSS_SECTIONAL_STD"),
        (
            "pb_percentile_3y",
            "rolling_percentile(pb_median, 756)",
            756,
            "ROLLING_EMPIRICAL_CDF",
        ),
        (
            "pb_percentile_5y",
            "rolling_percentile(pb_median, 1260)",
            1260,
            "ROLLING_EMPIRICAL_CDF",
        ),
        (
            "pb_percentile_expanding",
            "expanding_percentile(pb_median)",
            252,
            "EXPANDING_EMPIRICAL_CDF",
        ),
        ("pb_change_20d", "pb_median / delay(pb_median, 20) - 1", 20, "CHANGE"),
    )
    for prefix, scope, universe in scopes:
        for suffix, formula, lookback, aggregation in metrics:
            state_id = f"{prefix}_{suffix}"
            if state_id in existing:
                continue
            if prefix in {"growth", "value", "dividend"} and suffix not in {
                "pb_median",
                "aggregate_pb",
                "pb_percentile_3y",
            }:
                continue
            if prefix == "all_a" and suffix in {"pb_25pct", "pb_75pct", "pb_change_20d"}:
                continue
            specs.append(
                _spec(
                    state_id=state_id,
                    family=MarketStateFamily.VALUATION,
                    subfamily="pb",
                    scope=scope,
                    formula=formula,
                    lookback=lookback,
                    aggregation=aggregation,
                    interpretation="Higher values indicate richer price-to-book valuation.",
                    universe=universe,
                    role=MarketStateRole.VALUATION_STATE,
                    datasets=("daily_basic", "index_weight"),
                )
            )
    return tuple(specs)


def risk_appetite_state_specs() -> tuple[MarketStateSpec, ...]:
    specs: list[MarketStateSpec] = []
    pairs = (
        ("microcap_minus_largecap", "MICROCAP_DYNAMIC", "LARGECAP_DYNAMIC", True),
        ("high_vol_minus_low_vol", "HIGH_VOL_DYNAMIC", "LOW_VOL_DYNAMIC", True),
        ("low_price_minus_high_price", "LOW_PRICE_DYNAMIC", "HIGH_PRICE_DYNAMIC", True),
        ("high_turnover_minus_low_turnover", "HIGH_TURNOVER_DYNAMIC", "LOW_TURNOVER_DYNAMIC", True),
        ("growth_minus_dividend", "GROWTH_DYNAMIC", "DIVIDEND_DYNAMIC", True),
        ("high_beta_minus_low_beta", "HIGH_BETA_DYNAMIC", "LOW_BETA_DYNAMIC", False),
        ("loss_making_minus_profitable", "LOSS_MAKING_DYNAMIC", "PROFITABLE_DYNAMIC", False),
        ("technology_minus_dividend", "TECHNOLOGY", "DIVIDEND_DYNAMIC", False),
    )
    for prefix, left, right, available in pairs:
        for window in (20, 40, 60):
            specs.append(
                _spec(
                    state_id=f"{prefix}_return_{window}d",
                    family=MarketStateFamily.RISK_APPETITE,
                    subfamily="relative_portfolio_return",
                    scope=MarketStateScope.ALL_A,
                    formula=(
                        f"relative_return({left.lower()}, {right.lower()}, {window})"
                        if available
                        else f"unavailable({prefix}, missing_required_dataset)"
                    ),
                    lookback=window,
                    aggregation="COMPOUNDED_RELATIVE_RETURN",
                    interpretation="Positive values indicate stronger speculative risk appetite.",
                    universe="ALL_A",
                    role=MarketStateRole.RISK_STATE,
                    status=(
                        MarketStateStatus.IMPLEMENTED
                        if available
                        else MarketStateStatus.DATA_DEPENDENCY_MISSING
                    ),
                    parameters={"window": window, "left": left, "right": right},
                    datasets=(
                        ("daily", "daily_basic", "fina_indicator")
                        if available
                        else ("missing_required_dataset",)
                    ),
                )
            )
    return tuple(specs)


def fundamental_state_specs() -> tuple[MarketStateSpec, ...]:
    specs: list[MarketStateSpec] = []
    scopes = (
        ("all_a", MarketStateScope.ALL_A, "ALL_A"),
        ("growth", MarketStateScope.GROWTH, "GROWTH_DYNAMIC"),
        ("value", MarketStateScope.VALUE, "VALUE_DYNAMIC"),
        ("microcap", MarketStateScope.MICROCAP, "MICROCAP_FIXED_MONTHLY"),
        ("dividend", MarketStateScope.DIVIDEND, "DIVIDEND_DYNAMIC"),
    )
    for prefix, scope, universe in scopes:
        for suffix, formula, interpretation in (
            ("profit_growth_median", "median(netprofit_yoy)", "Higher means faster profit growth."),
            (
                "negative_profit_growth_ratio",
                "mean(netprofit_yoy < 0)",
                "Higher means broader profit contraction.",
            ),
            (
                "debt_to_assets_median",
                "median(debt_to_assets)",
                "Higher means greater balance-sheet leverage.",
            ),
        ):
            specs.append(
                _spec(
                    state_id=f"{prefix}_{suffix}",
                    family=MarketStateFamily.FUNDAMENTALS,
                    subfamily=suffix,
                    scope=scope,
                    formula=formula,
                    lookback=1,
                    aggregation="CROSS_SECTIONAL_PIT_AGGREGATE",
                    interpretation=interpretation,
                    universe=universe,
                    role=MarketStateRole.REGIME_INPUT,
                    datasets=("fina_indicator",),
                )
            )
    for state_id, field in (
        ("all_a_revenue_growth_median", "revenue_growth"),
        ("all_a_roe_median", "roe"),
        ("all_a_loss_making_ratio", "net_profit"),
        ("revenue_growth_breadth", "revenue_growth"),
        ("roe_improvement_ratio", "roe"),
        ("profit_revision_proxy", "earnings_revision"),
        ("microcap_pe_percentile", "pe"),
        ("microcap_ep_percentile", "ep"),
        ("microcap_relative_pe_vs_all_a", "pe"),
        ("dividend_yield_spread_vs_10y_bond", "cn_10y_bond_yield"),
    ):
        specs.append(
            _spec(
                state_id=state_id,
                family=(
                    MarketStateFamily.VALUATION
                    if "pe" in state_id or "yield_spread" in state_id
                    else MarketStateFamily.FUNDAMENTALS
                ),
                subfamily="missing_source",
                scope=MarketStateScope.ALL_A,
                formula=f"unavailable({state_id}, missing({field}))",
                lookback=1,
                aggregation="UNAVAILABLE",
                interpretation=f"Requires {field}, which is absent from the current release.",
                universe="ALL_A",
                status=MarketStateStatus.DATA_DEPENDENCY_MISSING,
                datasets=(field,),
            )
        )
    return tuple(specs)


def extended_state_specs() -> tuple[MarketStateSpec, ...]:
    groups: Iterable[tuple[MarketStateSpec, ...]] = (
        trend_state_specs(),
        breadth_state_specs(),
        valuation_extension_specs(),
        risk_appetite_state_specs(),
        fundamental_state_specs(),
    )
    return tuple(spec for group in groups for spec in group)
