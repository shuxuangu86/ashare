import numpy as np

from aquant.factors.atomic.builders import (
    amihud_factor,
    days_from_high_factor,
    distance_to_high_factor,
    momentum_factor,
    point_factor,
    reversal_factor,
    rolling_change_factor,
    rolling_correlation_factor,
    rolling_mean_factor,
    trend_slope_factor,
    volatility_factor,
    zero_return_days_factor,
)
from aquant.factors.atomic.models import AtomicFactor, FactorPanelInput, atomic_factor
from aquant.factors.operators.cross_sectional import cs_percentile
from aquant.factors.operators.math import log, safe_div
from aquant.factors.operators.time_series import (
    count_if,
    delay,
    delta,
    returns,
    rolling_corr,
    rolling_max,
    rolling_mean,
    rolling_skew,
    rolling_std,
    rolling_sum,
)


def _field(panel: FactorPanelInput, name: str) -> np.ndarray:  # type: ignore[type-arg]
    return np.asarray(panel.fields[name], dtype=np.float64)


def _custom(
    factor_id: str,
    family: str,
    fields: tuple[str, ...],
    history: int,
    calculator: object,
    description: str,
    hypothesis: str,
    *,
    direction: int = 0,
    parameters: dict[str, object] | None = None,
) -> AtomicFactor:
    return atomic_factor(
        factor_id,
        family,
        description,
        hypothesis,
        fields,
        history,
        calculator,  # type: ignore[arg-type]
        expected_direction=direction,
        parameters=parameters,
    )


def _size_factors() -> tuple[AtomicFactor, ...]:
    return (
        point_factor(
            "log_total_market_cap",
            "size",
            "total_market_cap",
            log,
            "Natural logarithm of total market capitalization.",
            "Firm size is a persistent cross-sectional risk and return exposure.",
            direction=-1,
        ),
        point_factor(
            "log_float_market_cap",
            "size",
            "float_market_cap",
            log,
            "Natural logarithm of float market capitalization.",
            "Tradable size isolates the investable scale exposure.",
            direction=-1,
        ),
        point_factor(
            "price_level",
            "size",
            "close",
            log,
            "Natural logarithm of closing price.",
            "Nominal price captures retail preference and trading constraints.",
            direction=-1,
        ),
        _custom(
            "float_cap_ratio",
            "size",
            ("float_market_cap", "total_market_cap"),
            1,
            lambda p: safe_div(_field(p, "float_market_cap"), _field(p, "total_market_cap")),
            "Float capitalization divided by total capitalization.",
            "A low float fraction may amplify supply-demand imbalances.",
        ),
    )


def _fundamental_factors() -> tuple[AtomicFactor, ...]:
    return (
        point_factor(
            "book_to_market",
            "value",
            "pb",
            lambda x: safe_div(1, x),
            "Book-to-market ratio derived from PIT price-to-book.",
            "Cheaper firms may earn a value premium.",
            direction=1,
        ),
        point_factor(
            "dividend_yield",
            "value",
            "dividend_yield",
            lambda x: x,
            "PIT trailing dividend yield.",
            "Cash distributions support the value signal.",
            direction=1,
        ),
        point_factor(
            "leverage",
            "quality",
            "debt_to_assets",
            lambda x: x,
            "PIT debt-to-assets ratio.",
            "High balance-sheet leverage raises downside risk.",
            direction=-1,
        ),
        point_factor(
            "inverse_leverage",
            "quality",
            "debt_to_assets",
            lambda x: -x,
            "Negative PIT debt-to-assets ratio.",
            "Conservative balance sheets may exhibit higher quality.",
            direction=1,
        ),
        rolling_change_factor(
            "leverage_change_20d",
            "quality",
            "debt_to_assets",
            20,
            "Twenty-session change in PIT leverage.",
            "Rapid increases in leverage may signal deteriorating quality.",
            direction=-1,
        ),
        _custom(
            "profit_growth_stability_60d",
            "quality",
            ("netprofit_yoy",),
            60,
            lambda p: -rolling_std(_field(p, "netprofit_yoy"), 60),
            "Negative rolling variation in PIT profit growth.",
            "Stable reported growth is a quality characteristic.",
            direction=1,
            parameters={"window": 60},
        ),
        point_factor(
            "net_profit_growth_yoy",
            "growth",
            "netprofit_yoy",
            lambda x: x,
            "Latest PIT year-on-year net-profit growth.",
            "Profit growth may predict medium-horizon repricing.",
            direction=1,
        ),
        rolling_change_factor(
            "net_profit_growth_change_20d",
            "growth",
            "netprofit_yoy",
            20,
            "Twenty-session change in PIT year-on-year profit growth.",
            "Upward profit revisions may precede price adjustment.",
            direction=1,
        ),
        _custom(
            "net_profit_growth_acceleration_60d",
            "growth",
            ("netprofit_yoy",),
            61,
            lambda p: delta(delta(_field(p, "netprofit_yoy"), 20), 40),
            "Change in 20-session profit-growth revision versus 40 sessions earlier.",
            "Accelerating revisions can distinguish durable from stale growth.",
            direction=1,
            parameters={"short_window": 20, "comparison_lag": 40},
        ),
    )


def _momentum_reversal_factors() -> tuple[AtomicFactor, ...]:
    momentum = tuple(momentum_factor(window) for window in (5, 10, 20, 40, 60, 120, 250))
    reversal = tuple(reversal_factor(window) for window in (1, 3, 5, 10, 20))
    extras = (
        distance_to_high_factor(250),
        trend_slope_factor(20),
        trend_slope_factor(60),
        _custom(
            "momentum_12_1",
            "momentum",
            ("close",),
            251,
            lambda p: returns(_field(p, "close"), 250) - returns(_field(p, "close"), 20),
            "Twelve-month return excluding the latest month.",
            "Skipping the latest month reduces short-term reversal contamination.",
            direction=1,
            parameters={"long_window": 250, "skip_window": 20},
        ),
        _custom(
            "overnight_reversal",
            "reversal",
            ("open", "close"),
            2,
            lambda p: -safe_div(_field(p, "open"), delay(_field(p, "close"))) + 1,
            "Negative close-to-next-open return.",
            "Overnight price pressure may reverse.",
            direction=1,
        ),
        _custom(
            "intraday_reversal",
            "reversal",
            ("open", "close"),
            1,
            lambda p: -safe_div(_field(p, "close"), _field(p, "open")) + 1,
            "Negative open-to-close return.",
            "Intraday demand shocks may mean-revert.",
            direction=1,
        ),
    )
    return (*momentum, *reversal, *extras)


def _volatility_factors() -> tuple[AtomicFactor, ...]:
    standard = tuple(volatility_factor(window) for window in (10, 20, 40, 60, 120))
    return (
        *standard,
        _custom(
            "downside_volatility_20d",
            "volatility",
            ("close",),
            21,
            lambda p: rolling_std(np.minimum(returns(_field(p, "close")), 0), 20),
            "Twenty-session downside return volatility.",
            "Downside variation measures adverse rather than symmetric risk.",
            direction=-1,
            parameters={"window": 20},
        ),
        _custom(
            "downside_volatility_60d",
            "volatility",
            ("close",),
            61,
            lambda p: rolling_std(np.minimum(returns(_field(p, "close")), 0), 60),
            "Sixty-session downside return volatility.",
            "Persistent downside variation is an adverse risk exposure.",
            direction=-1,
            parameters={"window": 60},
        ),
        _custom(
            "max_daily_return_20d",
            "volatility",
            ("close",),
            21,
            lambda p: rolling_max(returns(_field(p, "close")), 20),
            "Maximum daily return over twenty sessions.",
            "Extreme recent gains proxy lottery-like demand.",
            direction=-1,
            parameters={"window": 20},
        ),
        _custom(
            "return_skewness_60d",
            "volatility",
            ("close",),
            61,
            lambda p: rolling_skew(returns(_field(p, "close")), 60),
            "Sixty-session daily-return skewness.",
            "Positive skew can indicate lottery-like payoff preference.",
            direction=-1,
            parameters={"window": 60},
        ),
        _custom(
            "high_low_range_volatility_20d",
            "volatility",
            ("high", "low", "close"),
            20,
            lambda p: rolling_std(
                safe_div(_field(p, "high") - _field(p, "low"), _field(p, "close")), 20
            ),
            "Variation of the intraday high-low range over twenty sessions.",
            "Unstable intraday ranges indicate execution and price risk.",
            direction=-1,
            parameters={"window": 20},
        ),
    )


def _liquidity_factors() -> tuple[AtomicFactor, ...]:
    return (
        *(
            rolling_mean_factor(
                f"turnover_{window}d",
                "liquidity",
                "turnover_rate",
                window,
                f"Mean turnover rate over {window} sessions.",
                "Turnover measures tradability and investor attention.",
                direction=1,
            )
            for window in (5, 20, 60)
        ),
        rolling_change_factor(
            "turnover_change_20d",
            "liquidity",
            "turnover_rate",
            20,
            "Twenty-session change in turnover rate.",
            "Rising turnover may capture a change in market attention.",
        ),
        _custom(
            "turnover_volatility",
            "liquidity",
            ("turnover_rate",),
            20,
            lambda p: rolling_std(_field(p, "turnover_rate"), 20),
            "Twenty-session turnover-rate volatility.",
            "Unstable turnover indicates less reliable capacity.",
            direction=-1,
            parameters={"window": 20},
        ),
        *(
            rolling_mean_factor(
                f"amount_{window}d",
                "liquidity",
                "amount",
                window,
                f"Mean traded amount over {window} sessions.",
                "Traded amount is a direct capacity proxy.",
                direction=1,
            )
            for window in (5, 20, 60)
        ),
        amihud_factor(20),
        zero_return_days_factor(20),
        _custom(
            "volume_shock",
            "liquidity",
            ("volume",),
            21,
            lambda p: safe_div(_field(p, "volume"), rolling_mean(_field(p, "volume"), 20)) - 1,
            "Current volume relative to its twenty-session mean.",
            "Unusual volume signals an attention or information shock.",
            parameters={"window": 20},
        ),
        _custom(
            "amount_shock",
            "liquidity",
            ("amount",),
            21,
            lambda p: safe_div(_field(p, "amount"), rolling_mean(_field(p, "amount"), 20)) - 1,
            "Current amount relative to its twenty-session mean.",
            "Unusual traded value signals a liquidity regime change.",
            parameters={"window": 20},
        ),
    )


def _price_volume_factors() -> tuple[AtomicFactor, ...]:
    def volume_ratio(panel: FactorPanelInput, positive: bool) -> np.ndarray:  # type: ignore[type-arg]
        daily_return = returns(_field(panel, "close"))
        selected = np.where((daily_return > 0) == positive, _field(panel, "volume"), 0)
        return safe_div(rolling_sum(selected, 20), rolling_sum(_field(panel, "volume"), 20))

    return (
        rolling_correlation_factor(
            "price_volume_corr_20d",
            "price_volume",
            "close",
            "volume",
            20,
            "Twenty-session correlation of close and volume.",
            "Price-volume confirmation distinguishes supported trends.",
        ),
        _custom(
            "return_volume_corr_20d",
            "price_volume",
            ("close", "volume"),
            21,
            lambda p: rolling_corr(returns(_field(p, "close")), _field(p, "volume"), 20),
            "Twenty-session correlation of return and volume.",
            "Return-volume dependence captures informed trading pressure.",
            parameters={"window": 20},
        ),
        _custom(
            "up_day_volume_ratio",
            "price_volume",
            ("close", "volume"),
            21,
            lambda p: volume_ratio(p, True),
            "Share of volume occurring on positive-return days.",
            "Volume concentrated on advances indicates demand confirmation.",
            direction=1,
            parameters={"window": 20},
        ),
        _custom(
            "down_day_volume_ratio",
            "price_volume",
            ("close", "volume"),
            21,
            lambda p: volume_ratio(p, False),
            "Share of volume occurring on non-positive-return days.",
            "Volume concentrated on declines indicates distribution pressure.",
            direction=-1,
            parameters={"window": 20},
        ),
        _custom(
            "close_location_value",
            "price_volume",
            ("high", "low", "close"),
            1,
            lambda p: safe_div(
                2 * _field(p, "close") - _field(p, "high") - _field(p, "low"),
                _field(p, "high") - _field(p, "low"),
            ),
            "Close location inside the daily high-low range.",
            "A close near the high indicates persistent intraday demand.",
            direction=1,
        ),
        _custom(
            "high_low_range",
            "price_volume",
            ("high", "low", "close"),
            1,
            lambda p: safe_div(_field(p, "high") - _field(p, "low"), _field(p, "close")),
            "Daily high-low range scaled by close.",
            "Wide ranges proxy uncertainty and adverse selection.",
            direction=-1,
        ),
        _custom(
            "gap_return",
            "price_volume",
            ("open", "close"),
            2,
            lambda p: safe_div(_field(p, "open"), delay(_field(p, "close"))) - 1,
            "Close-to-next-open gap return.",
            "Opening gaps isolate overnight information.",
        ),
        _custom(
            "vwap_deviation",
            "price_volume",
            ("close", "amount", "volume"),
            1,
            lambda p: safe_div(
                _field(p, "close") - safe_div(_field(p, "amount"), _field(p, "volume")),
                safe_div(_field(p, "amount"), _field(p, "volume")),
            ),
            "Close deviation from daily amount-weighted average price.",
            "Closing pressure relative to VWAP captures late-session demand.",
            parameters={
                "vwap_derivation": "amount_cny / volume_shares",
                "amount_unit": "CNY",
                "volume_unit": "shares",
                "unit_validation": "standardized bars_1d publisher contract",
            },
        ),
        _custom(
            "volume_price_divergence",
            "price_volume",
            ("close", "volume"),
            21,
            lambda p: (
                returns(_field(p, "close"), 20)
                - safe_div(_field(p, "volume"), rolling_mean(_field(p, "volume"), 20))
                + 1
            ),
            "Twenty-session return minus relative volume.",
            "Price moves unsupported by volume may be less persistent.",
        ),
        _custom(
            "trend_consistency",
            "price_volume",
            ("close",),
            21,
            lambda p: safe_div(
                count_if(returns(_field(p, "close")) > 0, 20),
                20,
            ),
            "Fraction of positive-return sessions in the last twenty days.",
            "Broadly distributed gains indicate a consistent trend.",
            direction=1,
            parameters={"window": 20},
        ),
        _custom(
            "positive_return_ratio",
            "price_volume",
            ("close",),
            61,
            lambda p: safe_div(count_if(returns(_field(p, "close")) > 0, 60), 60),
            "Fraction of positive-return sessions in the last sixty days.",
            "Persistent breadth of positive returns supports momentum.",
            direction=1,
            parameters={"window": 60},
        ),
        _custom(
            "limit_up_frequency_20d",
            "price_volume",
            ("close", "up_limit"),
            20,
            lambda p: safe_div(
                rolling_sum(np.isclose(_field(p, "close"), _field(p, "up_limit")), 20), 20
            ),
            "Frequency of closes at the upper price limit.",
            "Repeated limit closes capture crowding and execution risk.",
        ),
        days_from_high_factor(20),
    )


def _microcap_risk_factors() -> tuple[AtomicFactor, ...]:
    return (
        _custom(
            "microcap_score",
            "microcap_risk",
            ("float_market_cap",),
            1,
            lambda p: 1 - cs_percentile(_field(p, "float_market_cap")),
            "Inverse cross-sectional percentile of float market cap.",
            "Very small tradable capitalization identifies micro-cap exposure.",
            direction=1,
        ),
        point_factor(
            "low_price_risk",
            "microcap_risk",
            "close",
            lambda x: -log(x),
            "Negative logarithm of close price.",
            "Low nominal prices are associated with trading and delisting risk.",
            direction=-1,
        ),
        point_factor(
            "low_liquidity_risk",
            "microcap_risk",
            "turnover_rate",
            lambda x: -x,
            "Negative current turnover rate.",
            "Weak turnover increases implementation risk.",
            direction=-1,
        ),
        _custom(
            "limit_up_buy_risk",
            "microcap_risk",
            ("close", "up_limit"),
            1,
            lambda p: 1 - safe_div(_field(p, "up_limit") - _field(p, "close"), _field(p, "close")),
            "Proximity to the upper price limit.",
            "Near-limit securities may be impossible to buy.",
            direction=-1,
        ),
        _custom(
            "limit_down_sell_risk",
            "microcap_risk",
            ("close", "down_limit"),
            1,
            lambda p: (
                1 - safe_div(_field(p, "close") - _field(p, "down_limit"), _field(p, "close"))
            ),
            "Proximity to the lower price limit.",
            "Near-limit securities may be impossible to sell.",
            direction=-1,
        ),
        _custom(
            "microcap_crowding_proxy",
            "microcap_risk",
            ("turnover_rate", "float_market_cap"),
            1,
            lambda p: safe_div(_field(p, "turnover_rate"), log(_field(p, "float_market_cap"))),
            "Turnover scaled by logarithmic float capitalization.",
            "High activity in limited float can proxy micro-cap crowding.",
            direction=-1,
        ),
        _custom(
            "float_scarcity_risk",
            "microcap_risk",
            ("total_market_cap", "float_market_cap"),
            1,
            lambda p: safe_div(_field(p, "total_market_cap"), _field(p, "float_market_cap")),
            "Total capitalization divided by float capitalization.",
            "A scarce tradable float can amplify price impact.",
            direction=-1,
        ),
    )


def baseline_factor_library() -> tuple[AtomicFactor, ...]:
    factors = (
        *_size_factors(),
        *_fundamental_factors(),
        *_momentum_reversal_factors(),
        *_volatility_factors(),
        *_liquidity_factors(),
        *_price_volume_factors(),
        *_microcap_risk_factors(),
    )
    keys = [(factor.spec.factor_id, factor.spec.version) for factor in factors]
    if len(keys) != len(set(keys)):
        raise RuntimeError("baseline factor library contains duplicate versions")
    return factors
