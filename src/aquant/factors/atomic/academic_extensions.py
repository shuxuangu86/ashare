from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from aquant.factors.atomic.models import Array, AtomicFactor, FactorPanelInput
from aquant.factors.operators.cross_sectional import cs_percentile
from aquant.factors.operators.math import safe_div
from aquant.factors.operators.time_series import (
    delay,
    delta,
    ema,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    rolling_sum,
    sma_cn,
)
from aquant.factors.spec import (
    FactorLayer,
    FactorRole,
    FactorSpec,
    FactorStatus,
    ImplementationStatus,
    SourceFaithfulness,
    SourceType,
)

Calculator = Callable[[FactorPanelInput], Array]


@dataclass(frozen=True, slots=True)
class TechnicalRuleSearchSpace:
    source_trial_count: int
    controlled_trial_count: int
    families: tuple[str, ...]
    parameter_space: dict[str, Any]
    search_space_hash: str


def academic_extension_library() -> tuple[AtomicFactor, ...]:
    factors = (
        *han_yang_zhou_library(),
        *china_7000_controlled_library(),
        *technical_sentiment_library(),
    )
    ids = {factor.spec.factor_id for factor in factors}
    hashes = {factor.spec.expression_hash for factor in factors}
    if len(ids) != len(factors) or len(hashes) != len(factors):
        raise RuntimeError("academic extension library contains duplicate definitions")
    return factors


def han_yang_zhou_library() -> tuple[AtomicFactor, ...]:
    components = (
        _academic_factor(
            "hyz_ma_signal_5_20",
            "Moving-average signal, normalized 5-day versus 20-day spread.",
            "trend",
            "ma_component",
            ("close",),
            20,
            lambda panel: _ma_signal(panel, 5, 20),
            source_id="SRC_HAN_YANG_ZHOU_2013",
            formula_id="MA(5,20)",
            role=FactorRole.STATE_FEATURE,
            faithfulness=SourceFaithfulness.NORMALIZED_EQUIVALENT,
            parameters={"short_window": 5, "long_window": 20},
        ),
        _academic_factor(
            "hyz_realized_volatility_20",
            "Twenty-session volatility of close-to-close returns.",
            "volatility",
            "volatility_component",
            ("close",),
            21,
            lambda panel: rolling_std(_returns(_field(panel, "close")), 20),
            source_id="SRC_HAN_YANG_ZHOU_2013",
            formula_id="VOL20",
            role=FactorRole.RISK_FACTOR,
            faithfulness=SourceFaithfulness.NORMALIZED_EQUIVALENT,
            parameters={"window": 20},
        ),
        _academic_factor(
            "hyz_idiosyncratic_volatility_60",
            "Volatility of returns residualized against the equal-weight A-share market proxy.",
            "volatility",
            "uncertainty_component",
            ("close",),
            61,
            _idiosyncratic_volatility,
            source_id="SRC_HAN_YANG_ZHOU_2013",
            formula_id="IVOL60",
            role=FactorRole.RISK_FACTOR,
            faithfulness=SourceFaithfulness.A_SHARE_ADAPTED,
            parameters={"window": 60, "market_proxy": "equal_weight_all_a_share"},
        ),
        _academic_factor(
            "hyz_log_size",
            "Natural logarithm of PIT total market capitalization.",
            "size",
            "uncertainty_component",
            ("total_market_cap",),
            1,
            lambda panel: _safe_log(_field(panel, "total_market_cap")),
            source_id="SRC_HAN_YANG_ZHOU_2013",
            formula_id="SIZE",
            role=FactorRole.CONTROL_FEATURE,
            faithfulness=SourceFaithfulness.NORMALIZED_EQUIVALENT,
        ),
        _academic_factor(
            "hyz_liquidity_20",
            "Negative log Amihud illiquidity, so larger values denote greater liquidity.",
            "liquidity",
            "uncertainty_component",
            ("close", "amount"),
            21,
            _liquidity,
            source_id="SRC_HAN_YANG_ZHOU_2013",
            formula_id="LIQ20",
            role=FactorRole.CONTROL_FEATURE,
            faithfulness=SourceFaithfulness.A_SHARE_ADAPTED,
            parameters={"window": 20, "amount_unit": "CNY"},
        ),
        _academic_factor(
            "hyz_turnover_activity_20",
            "Twenty-session average log turnover activity.",
            "liquidity",
            "uncertainty_component",
            ("turnover_rate",),
            20,
            lambda panel: rolling_mean(np.log1p(_field(panel, "turnover_rate")), 20),
            source_id="SRC_HAN_YANG_ZHOU_2013",
            formula_id="TURNOVER20",
            role=FactorRole.CONTROL_FEATURE,
            faithfulness=SourceFaithfulness.A_SHARE_ADAPTED,
            parameters={"window": 20},
        ),
        _academic_factor(
            "hyz_information_uncertainty_proxy",
            "Cross-sectional composite of volatility, small size, and illiquidity.",
            "risk",
            "uncertainty_component",
            ("close", "amount", "total_market_cap"),
            21,
            _information_uncertainty,
            source_id="SRC_HAN_YANG_ZHOU_2013",
            formula_id="IU_PROXY",
            role=FactorRole.RISK_FACTOR,
            faithfulness=SourceFaithfulness.A_SHARE_ADAPTED,
            parameters={
                "components": [
                    "hyz_realized_volatility_20",
                    "hyz_log_size",
                    "hyz_liquidity_20",
                ]
            },
        ),
    )
    interactions = (
        _hyz_interaction(
            "ma_signal_x_volatility",
            "hyz_realized_volatility_20",
            ("close",),
            21,
            lambda panel: (
                _ma_signal(panel, 5, 20) * rolling_std(_returns(_field(panel, "close")), 20)
            ),
        ),
        _hyz_interaction(
            "ma_signal_x_idiosyncratic_volatility",
            "hyz_idiosyncratic_volatility_60",
            ("close",),
            61,
            lambda panel: _ma_signal(panel, 5, 20) * _idiosyncratic_volatility(panel),
        ),
        _hyz_interaction(
            "ma_signal_x_size",
            "hyz_log_size",
            ("close", "total_market_cap"),
            20,
            lambda panel: _ma_signal(panel, 5, 20) * _safe_log(_field(panel, "total_market_cap")),
        ),
        _hyz_interaction(
            "ma_signal_x_liquidity",
            "hyz_liquidity_20",
            ("close", "amount"),
            21,
            lambda panel: _ma_signal(panel, 5, 20) * _liquidity(panel),
        ),
        _hyz_interaction(
            "ma_signal_x_turnover",
            "hyz_turnover_activity_20",
            ("close", "turnover_rate"),
            20,
            lambda panel: (
                _ma_signal(panel, 5, 20)
                * rolling_mean(np.log1p(_field(panel, "turnover_rate")), 20)
            ),
        ),
        _hyz_interaction(
            "trend_x_information_uncertainty_proxy",
            "hyz_information_uncertainty_proxy",
            ("close", "amount", "total_market_cap"),
            21,
            lambda panel: _ma_signal(panel, 5, 20) * _information_uncertainty(panel),
        ),
    )
    return (*components, *interactions)


def china_rule_search_space() -> TechnicalRuleSearchSpace:
    short_windows = (1, 2, 5, 10, 20)
    long_windows = (20, 50, 100, 150, 200, 250)
    ma_pairs = tuple(
        (short, long) for short in short_windows for long in long_windows if short < long
    )
    space: dict[str, Any] = {
        "filter": {
            "threshold": (
                0.005,
                0.01,
                0.015,
                0.02,
                0.025,
                0.03,
                0.035,
                0.04,
                0.045,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                0.1,
                0.12,
                0.14,
                0.16,
                0.18,
                0.2,
            ),
            "holding_days": (1, 5, 10),
        },
        "moving_average": {
            "pairs": ma_pairs,
            "band": (0.0, 0.005, 0.01),
            "holding_days": (1, 5, 10),
        },
        "support_resistance": {
            "window": (5, 10, 20, 50, 100, 150, 200, 250),
            "band": (0.0, 0.005, 0.01, 0.02),
            "holding_days": (1, 5, 10),
        },
        "channel_breakout": {
            "window": (5, 10, 20, 50, 100, 150, 200, 250),
            "band": (0.0, 0.01, 0.05, 0.1),
            "holding_days": (1, 5, 10),
        },
        "obv_average": {
            "pairs": ma_pairs,
            "band": (0.0, 0.01),
            "holding_days": (1, 5, 10),
        },
    }
    controlled = (
        len(space["filter"]["threshold"]) * len(space["filter"]["holding_days"])
        + len(ma_pairs)
        * len(space["moving_average"]["band"])
        * len(space["moving_average"]["holding_days"])
        + len(space["support_resistance"]["window"])
        * len(space["support_resistance"]["band"])
        * len(space["support_resistance"]["holding_days"])
        + len(space["channel_breakout"]["window"])
        * len(space["channel_breakout"]["band"])
        * len(space["channel_breakout"]["holding_days"])
        + len(ma_pairs)
        * len(space["obv_average"]["band"])
        * len(space["obv_average"]["holding_days"])
    )
    serialized = json.dumps(space, sort_keys=True, separators=(",", ":"))
    return TechnicalRuleSearchSpace(
        source_trial_count=7_846,
        controlled_trial_count=controlled,
        families=tuple(space),
        parameter_space=space,
        search_space_hash=hashlib.sha256(serialized.encode()).hexdigest(),
    )


def china_7000_controlled_library() -> tuple[AtomicFactor, ...]:
    space = china_rule_search_space()
    factors: list[AtomicFactor] = []
    for threshold in space.parameter_space["filter"]["threshold"]:
        for holding_days in space.parameter_space["filter"]["holding_days"]:
            factors.append(_filter_rule_factor(float(threshold), int(holding_days), space))
    for short, long in space.parameter_space["moving_average"]["pairs"]:
        for band in space.parameter_space["moving_average"]["band"]:
            for holding_days in space.parameter_space["moving_average"]["holding_days"]:
                factors.append(
                    _ma_rule_factor(
                        int(short),
                        int(long),
                        float(band),
                        int(holding_days),
                        space,
                        obv=False,
                    )
                )
    for family, calculator in (
        ("support_resistance", _support_resistance_strength),
        ("channel_breakout", _channel_breakout_strength),
    ):
        parameters = space.parameter_space[family]
        for window in parameters["window"]:
            for band in parameters["band"]:
                for holding_days in parameters["holding_days"]:
                    factors.append(
                        _breakout_rule_factor(
                            family,
                            int(window),
                            float(band),
                            int(holding_days),
                            calculator,
                            space,
                        )
                    )
    for short, long in space.parameter_space["obv_average"]["pairs"]:
        for band in space.parameter_space["obv_average"]["band"]:
            for holding_days in space.parameter_space["obv_average"]["holding_days"]:
                factors.append(
                    _ma_rule_factor(
                        int(short),
                        int(long),
                        float(band),
                        int(holding_days),
                        space,
                        obv=True,
                    )
                )
    if len(factors) != space.controlled_trial_count:
        raise RuntimeError("controlled technical-rule count does not match its registry")
    return tuple(factors)


def technical_sentiment_library() -> tuple[AtomicFactor, ...]:
    definitions: tuple[tuple[str, str, FactorRole, Callable[[Array, Array, Array], Array]], ...] = (
        (
            "bullish_signal_count",
            "Count of valid canonical technical rules in a bullish state.",
            FactorRole.STATE_FEATURE,
            lambda signals, _trend, _oscillator: _signal_count(signals, positive=True),
        ),
        (
            "bearish_signal_count",
            "Count of valid canonical technical rules in a bearish state.",
            FactorRole.STATE_FEATURE,
            lambda signals, _trend, _oscillator: _signal_count(signals, positive=False),
        ),
        (
            "net_technical_sentiment",
            "Bullish minus bearish rules, divided by the number of valid rules.",
            FactorRole.ALPHA_CANDIDATE,
            lambda signals, _trend, _oscillator: _signal_mean(signals),
        ),
        (
            "signal_agreement",
            "Absolute normalized net technical sentiment.",
            FactorRole.STATE_FEATURE,
            lambda signals, _trend, _oscillator: np.abs(_signal_mean(signals)),
        ),
        (
            "signal_dispersion",
            "Cross-rule dispersion of canonical technical states.",
            FactorRole.STATE_FEATURE,
            lambda signals, _trend, _oscillator: _signal_dispersion(signals),
        ),
        (
            "trend_signal_breadth",
            "Mean state of canonical trend-following rules.",
            FactorRole.STATE_FEATURE,
            lambda _signals, trend, _oscillator: _signal_mean(trend),
        ),
        (
            "oscillator_signal_breadth",
            "Mean state of canonical oscillator rules.",
            FactorRole.STATE_FEATURE,
            lambda _signals, _trend, oscillator: _signal_mean(oscillator),
        ),
    )
    return tuple(
        _academic_factor(
            factor_id,
            description,
            "technical_sentiment",
            "signal_aggregation",
            ("close", "high", "low", "volume"),
            61,
            _sentiment_calculator(calculator),
            source_id="SRC_TECH_SENTIMENT_2023",
            formula_id=factor_id.upper(),
            role=role,
            faithfulness=SourceFaithfulness.A_SHARE_ADAPTED,
            parameters={
                "component_set": "aquant_canonical_10_rules_v1",
                "aggregation": factor_id,
                "source_rule_count": 2_127,
            },
            notes=(
                "Uses ten pre-declared daily rules; no performance weighting or "
                "test-period rule selection."
            ),
        )
        for factor_id, description, role, calculator in definitions
    )


def _academic_factor(
    factor_id: str,
    description: str,
    family: str,
    subfamily: str,
    fields: tuple[str, ...],
    history: int,
    calculator: Calculator,
    *,
    source_id: str,
    formula_id: str,
    role: FactorRole,
    faithfulness: SourceFaithfulness,
    parameters: dict[str, Any] | None = None,
    parent_ids: tuple[str, ...] = (),
    notes: str = "Daily close-derived values are usable from the next tradable session.",
) -> AtomicFactor:
    resolved_parameters = parameters or {}
    spec = FactorSpec(
        factor_id=factor_id,
        name=factor_id,
        description=description,
        family=family,
        subfamily=subfamily,
        role=role,
        layer=FactorLayer.L2B,
        version="1.0.0",
        status=FactorStatus.DRAFT,
        hypothesis=(
            "The published characteristic may contain complementary cross-sectional information."
        ),
        expected_direction=0,
        implementation=f"aquant.factors.atomic.academic_extensions:{factor_id}",
        input_fields=fields,
        parent_factor_ids=parent_ids,
        required_history=history,
        minimum_periods=max(1, history),
        data_lag=1,
        availability_lag=1,
        universe="all_a_share",
        target_horizons=(1, 5, 10, 20, 60),
        parameters=resolved_parameters,
        required_datasets=(
            ("bars_1d", "daily_basic_pit")
            if any(field in {"total_market_cap", "turnover_rate"} for field in fields)
            else ("bars_1d",)
        ),
        source_type=SourceType.ACADEMIC_PAPER,
        source_reference=source_id,
        source_id=source_id,
        source_section="factor construction",
        source_formula_id=formula_id,
        source_faithfulness=faithfulness,
        implementation_notes=notes,
        normalization="SOURCE_OR_DOCUMENTED_ADAPTATION",
        missing_policy="PRESERVE",
        warmup_policy="REQUIRE_MINIMUM_PERIODS",
        complexity_score=float(2 + len(fields) + len(resolved_parameters) + len(parent_ids)),
        implementation_status=ImplementationStatus.IMPLEMENTED,
        tags=(family, subfamily, "academic_extension_v1", "next_session_only"),
    )
    return AtomicFactor(spec, calculator)


def _hyz_interaction(
    factor_id: str,
    second_parent: str,
    fields: tuple[str, ...],
    history: int,
    calculator: Calculator,
) -> AtomicFactor:
    return _academic_factor(
        factor_id,
        f"Interaction between the moving-average state and {second_parent}.",
        "interaction",
        "technical_uncertainty",
        fields,
        history,
        calculator,
        source_id="SRC_HAN_YANG_ZHOU_2013",
        formula_id=factor_id.upper(),
        role=FactorRole.ALPHA_CANDIDATE,
        faithfulness=SourceFaithfulness.DERIVED_VARIANT,
        parent_ids=("hyz_ma_signal_5_20", second_parent),
        parameters={
            "component_1": "hyz_ma_signal_5_20",
            "component_2": second_parent,
            "interaction_order": 2,
        },
    )


def _rule_factor(
    factor_id: str,
    subfamily: str,
    fields: tuple[str, ...],
    history: int,
    calculator: Calculator,
    parameters: dict[str, Any],
    space: TechnicalRuleSearchSpace,
) -> AtomicFactor:
    return _academic_factor(
        factor_id,
        f"Continuous L2 representation of the {subfamily} technical rule.",
        "technical_rule",
        subfamily,
        fields,
        history,
        calculator,
        source_id="SRC_CHINA_7000_RULES",
        formula_id=subfamily.upper(),
        role=FactorRole.STATE_FEATURE,
        faithfulness=SourceFaithfulness.DERIVED_VARIANT,
        parameters={
            **parameters,
            "factor_pack": "china_7000_rules_controlled_v1",
            "source_trial_count": space.source_trial_count,
            "controlled_trial_count": space.controlled_trial_count,
            "search_space_hash": space.search_space_hash,
        },
        notes=(
            "Individual-stock continuous adaptation of the source index timing rule; "
            "all trials remain in the declared family-level search space."
        ),
    )


def _filter_rule_factor(
    threshold: float,
    holding_days: int,
    space: TechnicalRuleSearchSpace,
) -> AtomicFactor:
    token = _rate_token(threshold)
    return _rule_factor(
        f"cn_rule_filter_x{token}_hold{holding_days}",
        "filter",
        ("close",),
        2,
        lambda panel: _hold_signal(
            _filter_strength(_field(panel, "close"), threshold),
            holding_days,
        ),
        {"threshold": threshold, "holding_days": holding_days},
        space,
    )


def _ma_rule_factor(
    short: int,
    long: int,
    band: float,
    holding_days: int,
    space: TechnicalRuleSearchSpace,
    *,
    obv: bool,
) -> AtomicFactor:
    subfamily = "obv_average" if obv else "moving_average"
    token = _rate_token(band)
    fields = ("close", "volume") if obv else ("close",)

    def calculate(panel: FactorPanelInput) -> Array:
        close = _field(panel, "close")
        if obv:
            signed = np.sign(delta(close)) * _field(panel, "volume")
            base = np.cumsum(np.where(np.isfinite(signed), signed, 0), axis=0)
        else:
            base = close
        fast = rolling_mean(base, short)
        slow = rolling_mean(base, long)
        spread = safe_div(fast - slow, np.abs(slow))
        return _hold_signal(_band_signal(spread, band), holding_days)

    return _rule_factor(
        f"cn_rule_{'obvma' if obv else 'ma'}_s{short}_l{long}_b{token}_hold{holding_days}",
        subfamily,
        fields,
        long,
        calculate,
        {
            "short_window": short,
            "long_window": long,
            "band": band,
            "holding_days": holding_days,
        },
        space,
    )


def _breakout_rule_factor(
    subfamily: str,
    window: int,
    band: float,
    holding_days: int,
    calculator: Callable[[FactorPanelInput, int, float], Array],
    space: TechnicalRuleSearchSpace,
) -> AtomicFactor:
    token = _rate_token(band)
    return _rule_factor(
        f"cn_rule_{'sr' if subfamily == 'support_resistance' else 'channel'}"
        f"_w{window}_b{token}_hold{holding_days}",
        subfamily,
        ("close", "high", "low") if subfamily == "channel_breakout" else ("close",),
        window + 1,
        lambda panel: _hold_signal(calculator(panel, window, band), holding_days),
        {"window": window, "band": band, "holding_days": holding_days},
        space,
    )


def _field(panel: FactorPanelInput, name: str) -> Array:
    return np.asarray(panel.fields[name], dtype=np.float64)


def _returns(close: Array) -> Array:
    return safe_div(close, delay(close)) - 1


def _safe_log(values: Array) -> Array:
    with np.errstate(all="ignore"):
        result = np.log(np.where(values > 0, values, np.nan))
    return np.where(np.isfinite(result), result, np.nan)


def _ma_signal(panel: FactorPanelInput, short: int, long: int) -> Array:
    close = _field(panel, "close")
    return safe_div(rolling_mean(close, short) - rolling_mean(close, long), close)


def _market_returns(close: Array) -> Array:
    returns = _returns(close)
    counts = np.sum(np.isfinite(returns), axis=1)
    market = safe_div(np.nansum(returns, axis=1), counts)
    return np.broadcast_to(market[:, None], close.shape)


def _idiosyncratic_volatility(panel: FactorPanelInput) -> Array:
    close = _field(panel, "close")
    residual = _returns(close) - _market_returns(close)
    return rolling_std(residual, 60)


def _liquidity(panel: FactorPanelInput) -> Array:
    illiquidity = rolling_mean(
        safe_div(np.abs(_returns(_field(panel, "close"))), _field(panel, "amount")),
        20,
    )
    return -np.log1p(illiquidity * 100_000_000)


def _information_uncertainty(panel: FactorPanelInput) -> Array:
    volatility = rolling_std(_returns(_field(panel, "close")), 20)
    size = _safe_log(_field(panel, "total_market_cap"))
    liquidity = _liquidity(panel)
    return (
        cs_percentile(volatility) + (1 - cs_percentile(size)) + (1 - cs_percentile(liquidity))
    ) / 3


def _band_signal(values: Array, band: float) -> Array:
    finite = np.isfinite(values)
    strength = np.maximum(np.abs(values) - band, 0)
    result = np.sign(values) * strength
    return np.where(finite, result, np.nan)


def _hold_signal(values: Array, holding_days: int) -> Array:
    result = np.full(values.shape, np.nan)
    for column in range(values.shape[1]):
        state = 0.0
        age = holding_days
        for row in range(values.shape[0]):
            value = values[row, column]
            if not np.isfinite(value):
                continue
            if value != 0:
                state = value
                age = 0
            elif age < holding_days:
                age += 1
            else:
                state = 0
            result[row, column] = state
    return result


def _filter_strength(close: Array, threshold: float) -> Array:
    result = np.full(close.shape, np.nan)
    for column in range(close.shape[1]):
        state = 0
        high = np.nan
        low = np.nan
        for row in range(close.shape[0]):
            value = close[row, column]
            if not np.isfinite(value):
                continue
            if not np.isfinite(high):
                high = low = value
                result[row, column] = 0
                continue
            high = max(high, value)
            low = min(low, value)
            if state >= 0 and value <= high * (1 - threshold):
                state = -1
                low = value
            elif state <= 0 and value >= low * (1 + threshold):
                state = 1
                high = value
            reference = high if state < 0 else low
            result[row, column] = state * abs(value / reference - 1)
    return result


def _support_resistance_strength(
    panel: FactorPanelInput,
    window: int,
    band: float,
) -> Array:
    close = _field(panel, "close")
    resistance = delay(rolling_max(close, window))
    support = delay(rolling_min(close, window))
    upper = safe_div(close, resistance) - 1 - band
    lower = safe_div(support, close) - 1 - band
    return np.where(upper > 0, upper, np.where(lower > 0, -lower, 0))


def _channel_breakout_strength(
    panel: FactorPanelInput,
    window: int,
    band: float,
) -> Array:
    close = _field(panel, "close")
    upper_channel = delay(rolling_max(_field(panel, "high"), window))
    lower_channel = delay(rolling_min(_field(panel, "low"), window))
    upper = safe_div(close, upper_channel) - 1 - band
    lower = safe_div(lower_channel, close) - 1 - band
    return np.where(upper > 0, upper, np.where(lower > 0, -lower, 0))


def _rate_token(value: float) -> str:
    return f"{round(value * 10_000):04d}"


def _sentiment_calculator(
    aggregation: Callable[[Array, Array, Array], Array],
) -> Calculator:
    def calculate(panel: FactorPanelInput) -> Array:
        trend, oscillator = _canonical_technical_signals(panel)
        signals = np.concatenate((trend, oscillator), axis=0)
        return aggregation(signals, trend, oscillator)

    return calculate


def _canonical_technical_signals(panel: FactorPanelInput) -> tuple[Array, Array]:
    close = _field(panel, "close")
    high = _field(panel, "high")
    low = _field(panel, "low")
    volume = _field(panel, "volume")
    ma_5_20 = _three_way(_ma_signal(panel, 5, 20), 0, 0)
    ma_20_60 = _three_way(_ma_signal(panel, 20, 60), 0, 0)
    macd_line = ema(close, 12, min_periods=12) - ema(close, 26, min_periods=26)
    macd_hist = macd_line - ema(macd_line, 9, min_periods=9)
    macd = _three_way(macd_hist, 0, 0)
    upper = delay(rolling_max(high, 20))
    lower = delay(rolling_min(low, 20))
    donchian = np.where(close > upper, 1.0, np.where(close < lower, -1.0, 0.0))
    donchian = np.where(np.isfinite(upper) & np.isfinite(lower), donchian, np.nan)
    signed_volume = np.sign(delta(close)) * volume
    obv = np.cumsum(np.where(np.isfinite(signed_volume), signed_volume, 0), axis=0)
    obv_trend = _three_way(rolling_mean(obv, 5) - rolling_mean(obv, 20), 0, 0)

    change = delta(close)
    gain, loss = np.maximum(change, 0), np.maximum(-change, 0)
    rsi = 100 * safe_div(sma_cn(gain, 14), sma_cn(gain + loss, 14))
    rsi_signal = _three_way(rsi, 55, 45)
    rsv = 100 * safe_div(
        close - rolling_min(low, 9),
        rolling_max(high, 9) - rolling_min(low, 9),
    )
    k = sma_cn(rsv, 3)
    j = 3 * k - 2 * sma_cn(k, 3)
    kdj_signal = _three_way(j, 55, 45)
    up = rolling_sum(np.maximum(change, 0), 14)
    down = rolling_sum(np.maximum(-change, 0), 14)
    cmo = 100 * safe_div(up - down, up + down)
    cmo_signal = _three_way(cmo, 10, -10)
    average = rolling_mean(close, 20)
    bias = 100 * safe_div(close - average, average)
    bias_signal = _three_way(bias, 0, 0)
    boll = safe_div(close - average, rolling_std(close, 20, ddof=0))
    boll_signal = _three_way(boll, 1, -1)
    return (
        np.stack((ma_5_20, ma_20_60, macd, donchian, obv_trend)),
        np.stack((rsi_signal, kdj_signal, cmo_signal, bias_signal, boll_signal)),
    )


def _three_way(values: Array, upper: float, lower: float) -> Array:
    result = np.where(values > upper, 1.0, np.where(values < lower, -1.0, 0.0))
    return np.where(np.isfinite(values), result, np.nan)


def _signal_mean(signals: Array) -> Array:
    valid = np.sum(np.isfinite(signals), axis=0)
    return safe_div(np.nansum(signals, axis=0), valid)


def _signal_count(signals: Array, *, positive: bool) -> Array:
    valid = np.sum(np.isfinite(signals), axis=0)
    selected = signals > 0 if positive else signals < 0
    count = np.sum(selected, axis=0).astype(float)
    return np.where(valid > 0, count, np.nan)


def _signal_dispersion(signals: Array) -> Array:
    mean = _signal_mean(signals)
    valid = np.sum(np.isfinite(signals), axis=0)
    squared = np.nansum(np.square(signals - mean[None, :, :]), axis=0)
    return np.sqrt(safe_div(squared, valid))
