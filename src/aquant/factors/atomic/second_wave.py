import numpy as np

from aquant.factors.atomic.models import Array, AtomicFactor, FactorPanelInput, atomic_factor
from aquant.factors.operators.cross_sectional import cs_demean, cs_percentile
from aquant.factors.operators.math import log, safe_div
from aquant.factors.operators.time_series import (
    delta,
    regression_slope,
    returns,
    rolling_kurt,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_skew,
    rolling_std,
    rolling_sum,
)


def _field(panel: FactorPanelInput, name: str) -> Array:
    return np.asarray(panel.fields[name], dtype=np.float64)


def _daily_return(panel: FactorPanelInput) -> Array:
    return returns(_field(panel, "close"))


def _residual_return(panel: FactorPanelInput) -> Array:
    return cs_demean(_daily_return(panel))


def _candidate(
    factor_id: str,
    family: str,
    fields: tuple[str, ...],
    history: int,
    calculator: object,
    description: str,
    hypothesis: str,
    *,
    direction: int,
    theme: str,
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
        parameters={
            "candidate_batch": "second_wave_v1",
            "research_theme": theme,
            "lookback_sessions": history,
            "minimum_periods": history,
            "missing_policy": "preserve",
            "pit_dependencies": fields,
        },
        tags=(family, "candidate", "second_wave_v1", "pit_safe"),
        source_reference="AQuant second-wave candidate library v1",
    )


def second_wave_candidate_library() -> tuple[AtomicFactor, ...]:
    factors = (
        _candidate(
            "market_residual_momentum_20d",
            "momentum",
            ("close",),
            21,
            lambda p: rolling_sum(_residual_return(p), 20),
            "Twenty-session sum of returns after daily market demeaning.",
            "Stock-specific price strength may persist beyond the common market move.",
            direction=1,
            theme="residual_momentum",
        ),
        _candidate(
            "residual_reversal_5d",
            "reversal",
            ("close",),
            6,
            lambda p: -rolling_sum(_residual_return(p), 5),
            "Negative five-session market-demeaned return.",
            "Short-lived idiosyncratic price pressure may reverse.",
            direction=1,
            theme="residual_reversal",
        ),
        _candidate(
            "residual_volatility_20d",
            "volatility",
            ("close",),
            21,
            lambda p: rolling_std(_residual_return(p), 20),
            "Twenty-session volatility of market-demeaned daily returns.",
            "Lower stock-specific volatility may command a premium.",
            direction=-1,
            theme="residual_volatility",
        ),
        _candidate(
            "residual_tail_shape_60d",
            "volatility",
            ("close",),
            61,
            lambda p: (
                rolling_skew(_residual_return(p), 60) - 0.1 * rolling_kurt(_residual_return(p), 60)
            ),
            "Residual-return skewness penalized by excess tail thickness.",
            "Asymmetric idiosyncratic tails may affect required returns.",
            direction=0,
            theme="residual_tail_risk",
        ),
        _candidate(
            "quality_momentum_20d",
            "quality",
            ("close", "debt_to_assets"),
            21,
            lambda p: (
                returns(_field(p, "close"), 20)
                * safe_div(1.0, 1.0 + np.abs(_field(p, "debt_to_assets")))
            ),
            "Twenty-session momentum scaled by inverse absolute leverage.",
            "Price strength backed by a less leveraged balance sheet may be more durable.",
            direction=1,
            theme="quality_momentum",
        ),
        _candidate(
            "growth_quality",
            "growth",
            ("netprofit_yoy", "debt_to_assets"),
            1,
            lambda p: safe_div(
                _field(p, "netprofit_yoy"),
                1.0 + np.abs(_field(p, "debt_to_assets")),
            ),
            "PIT profit growth scaled by balance-sheet leverage.",
            "Growth requiring less leverage may be higher quality.",
            direction=1,
            theme="growth_quality",
        ),
        _candidate(
            "growth_to_value",
            "growth",
            ("netprofit_yoy", "pb"),
            1,
            lambda p: safe_div(_field(p, "netprofit_yoy"), 1.0 + np.abs(_field(p, "pb"))),
            "PIT profit growth relative to price-to-book valuation.",
            "Growth purchased at a lower valuation may reprice more favorably.",
            direction=1,
            theme="growth_valuation",
        ),
        _candidate(
            "growth_stability_120d",
            "growth",
            ("netprofit_yoy",),
            120,
            lambda p: -rolling_std(_field(p, "netprofit_yoy"), 120),
            "Negative 120-session variability of visible profit growth.",
            "Stable PIT growth is less likely to be a transient revision.",
            direction=1,
            theme="growth_stability",
        ),
        _candidate(
            "leverage_adjusted_value",
            "value",
            ("pb", "debt_to_assets"),
            1,
            lambda p: safe_div(
                safe_div(1.0, _field(p, "pb")),
                1.0 + np.abs(_field(p, "debt_to_assets")),
            ),
            "Book-to-market scaled by inverse leverage.",
            "A value discount unsupported by excessive leverage may be more investable.",
            direction=1,
            theme="quality_value",
        ),
        _candidate(
            "growth_revision_intensity_60d",
            "growth",
            ("netprofit_yoy",),
            61,
            lambda p: safe_div(
                np.abs(delta(_field(p, "netprofit_yoy"), 20)),
                1.0 + rolling_std(_field(p, "netprofit_yoy"), 60),
            ),
            "Magnitude of recent PIT growth revision relative to its trailing variability.",
            "Large standardized revisions identify information-rich updates.",
            direction=0,
            theme="revision_intensity",
        ),
        _candidate(
            "turnover_shock_60d",
            "liquidity",
            ("turnover_rate",),
            60,
            lambda p: safe_div(
                _field(p, "turnover_rate") - rolling_mean(_field(p, "turnover_rate"), 60),
                rolling_std(_field(p, "turnover_rate"), 60),
            ),
            "Current turnover deviation from its 60-session mean in standard-deviation units.",
            "Unusual turnover captures a change in attention and trading demand.",
            direction=0,
            theme="turnover_shock",
        ),
        _candidate(
            "liquidity_dry_up_5_60",
            "liquidity",
            ("amount",),
            60,
            lambda p: (
                -safe_div(
                    rolling_mean(_field(p, "amount"), 5),
                    rolling_mean(_field(p, "amount"), 60),
                )
            ),
            "Negative ratio of five-session to 60-session traded amount.",
            "A recent liquidity dry-up raises implementation risk.",
            direction=1,
            theme="liquidity_dry_up",
        ),
        _candidate(
            "downside_amihud_20d",
            "liquidity",
            ("close", "amount"),
            21,
            lambda p: rolling_mean(
                safe_div(
                    np.where(_daily_return(p) < 0, np.abs(_daily_return(p)), 0.0),
                    _field(p, "amount"),
                ),
                20,
            ),
            "Mean downside absolute return per traded amount.",
            "Price impact concentrated on down days indicates adverse liquidity.",
            direction=-1,
            theme="downside_liquidity",
        ),
        _candidate(
            "volume_concentration_20d",
            "liquidity",
            ("volume",),
            20,
            lambda p: safe_div(
                rolling_max(_field(p, "volume"), 20),
                rolling_sum(_field(p, "volume"), 20),
            ),
            "Largest daily volume share within twenty sessions.",
            "Concentrated volume makes observed liquidity less persistent.",
            direction=-1,
            theme="volume_concentration",
        ),
        _candidate(
            "amount_concentration_20d",
            "liquidity",
            ("amount",),
            20,
            lambda p: safe_div(
                rolling_max(_field(p, "amount"), 20),
                rolling_sum(_field(p, "amount"), 20),
            ),
            "Largest daily traded-amount share within twenty sessions.",
            "Capacity dominated by one session is less reliable.",
            direction=-1,
            theme="amount_concentration",
        ),
        _candidate(
            "turnover_trend_20d",
            "liquidity",
            ("turnover_rate",),
            20,
            lambda p: regression_slope(log(1.0 + _field(p, "turnover_rate")), 20),
            "Twenty-session slope of log one plus turnover.",
            "A persistent turnover trend distinguishes attention from a one-day shock.",
            direction=0,
            theme="turnover_trend",
        ),
        _candidate(
            "upside_downside_vol_ratio_20d",
            "volatility",
            ("close",),
            21,
            lambda p: safe_div(
                rolling_std(np.maximum(_daily_return(p), 0.0), 20),
                rolling_std(np.minimum(_daily_return(p), 0.0), 20),
            ),
            "Ratio of upside to downside realized volatility.",
            "Return asymmetry differentiates favorable variability from downside risk.",
            direction=1,
            theme="tail_asymmetry",
        ),
        _candidate(
            "tail_return_asymmetry_60d",
            "volatility",
            ("close",),
            61,
            lambda p: rolling_max(_daily_return(p), 60) + rolling_min(_daily_return(p), 60),
            "Sum of the largest gain and largest loss over sixty sessions.",
            "The balance of extreme gains and losses captures tail asymmetry.",
            direction=0,
            theme="tail_asymmetry",
        ),
        _candidate(
            "jump_proxy_20d",
            "volatility",
            ("close",),
            21,
            lambda p: safe_div(
                rolling_max(np.abs(_daily_return(p)), 20),
                rolling_std(_daily_return(p), 20),
            ),
            "Largest absolute return relative to realized volatility.",
            "Jump-dominated volatility is less diversifiable than diffuse volatility.",
            direction=-1,
            theme="jump_risk",
        ),
        _candidate(
            "drawdown_recovery_60d",
            "volatility",
            ("close",),
            60,
            lambda p: safe_div(_field(p, "close"), rolling_min(_field(p, "close"), 60)) - 1.0,
            "Recovery from the trailing 60-session closing-price low.",
            "A sustained recovery may distinguish resilience from an unresolved drawdown.",
            direction=1,
            theme="drawdown_recovery",
        ),
        _candidate(
            "realized_range_20d",
            "volatility",
            ("high", "low", "close"),
            20,
            lambda p: rolling_mean(
                safe_div(_field(p, "high") - _field(p, "low"), _field(p, "close")),
                20,
            ),
            "Mean close-scaled high-low range over twenty sessions.",
            "Persistent intraday range is a robust daily-data risk proxy.",
            direction=-1,
            theme="realized_range",
        ),
        _candidate(
            "microcap_liquidity_quality",
            "microcap_risk",
            ("float_market_cap", "amount"),
            20,
            lambda p: (
                (1.0 - cs_percentile(_field(p, "float_market_cap")))
                * cs_percentile(rolling_mean(_field(p, "amount"), 20))
            ),
            "Micro-cap exposure interacted with cross-sectional traded-amount quality.",
            "Liquid micro-caps may retain the size premium with lower implementation risk.",
            direction=1,
            theme="microcap_quality",
        ),
        _candidate(
            "microcap_momentum_20d",
            "microcap_risk",
            ("float_market_cap", "close"),
            21,
            lambda p: (
                (1.0 - cs_percentile(_field(p, "float_market_cap")))
                * cs_percentile(returns(_field(p, "close"), 20))
            ),
            "Micro-cap exposure interacted with twenty-session momentum.",
            "Momentum may identify less fragile names within the micro-cap segment.",
            direction=1,
            theme="microcap_momentum",
        ),
        _candidate(
            "microcap_growth_quality",
            "microcap_risk",
            ("float_market_cap", "netprofit_yoy", "debt_to_assets"),
            1,
            lambda p: (
                (1.0 - cs_percentile(_field(p, "float_market_cap")))
                * safe_div(
                    _field(p, "netprofit_yoy"),
                    1.0 + np.abs(_field(p, "debt_to_assets")),
                )
            ),
            "Micro-cap exposure interacted with leverage-adjusted PIT growth.",
            "Fundamental quality may separate investable micro-caps from distressed size.",
            direction=1,
            theme="microcap_growth",
        ),
        _candidate(
            "volume_weighted_return_pressure_20d",
            "price_volume",
            ("close", "volume"),
            21,
            lambda p: safe_div(
                rolling_sum(_daily_return(p) * _field(p, "volume"), 20),
                rolling_sum(_field(p, "volume"), 20),
            ),
            "Twenty-session volume-weighted daily return pressure.",
            "Returns confirmed by volume may contain more persistent information.",
            direction=1,
            theme="volume_return_pressure",
        ),
        _candidate(
            "downside_volume_pressure_20d",
            "price_volume",
            ("close", "volume"),
            21,
            lambda p: safe_div(
                rolling_sum(
                    np.where(_daily_return(p) < 0, _field(p, "volume"), 0.0),
                    20,
                ),
                rolling_sum(_field(p, "volume"), 20),
            ),
            "Share of volume occurring on negative-return days over twenty sessions.",
            "Persistent downside volume indicates distribution pressure.",
            direction=-1,
            theme="downside_volume",
        ),
        _candidate(
            "range_contraction_5_20",
            "price_volume",
            ("high", "low", "close"),
            20,
            lambda p: (
                -safe_div(
                    rolling_mean(
                        safe_div(_field(p, "high") - _field(p, "low"), _field(p, "close")),
                        5,
                    ),
                    rolling_mean(
                        safe_div(_field(p, "high") - _field(p, "low"), _field(p, "close")),
                        20,
                    ),
                )
            ),
            "Negative ratio of five-session to twenty-session realized range.",
            "Range contraction can identify orderly consolidation rather than instability.",
            direction=1,
            theme="range_contraction",
        ),
        _candidate(
            "turnover_adjusted_momentum_20d",
            "momentum",
            ("close", "turnover_rate"),
            21,
            lambda p: safe_div(
                returns(_field(p, "close"), 20),
                1.0 + rolling_mean(_field(p, "turnover_rate"), 20),
            ),
            "Twenty-session momentum scaled by average turnover.",
            "Price strength requiring less turnover may be more efficient and persistent.",
            direction=1,
            theme="tradable_momentum",
        ),
    )
    ids = [factor.spec.factor_id for factor in factors]
    if len(ids) != len(set(ids)):
        raise RuntimeError("second-wave candidate library contains duplicate ids")
    return factors
