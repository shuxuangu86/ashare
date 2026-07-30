from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from aquant.factors.atomic.models import Array, AtomicFactor, FactorPanelInput
from aquant.factors.operators.math import safe_div
from aquant.factors.operators.time_series import (
    delay,
    delta,
    ema,
    regression_slope,
    rolling_corr,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_rank,
    rolling_std,
    rolling_sum,
    sma_cn,
    ts_zscore,
    wma,
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


class TechnicalTransformation(StrEnum):
    LEVEL = "level"
    NORMALIZED_LEVEL = "normalized_level"
    DELTA = "delta"
    SLOPE = "slope"
    ACCELERATION = "acceleration"
    TS_PERCENTILE = "ts_percentile"
    ZSCORE = "zscore"
    THRESHOLD_DISTANCE = "threshold_distance"
    CROSS_STATE = "cross_state"
    STATE_DURATION = "state_duration"
    RECOVERY_FROM_EXTREME = "recovery_from_extreme"
    DIVERGENCE_WITH_PRICE = "divergence_with_price"
    DIVERGENCE_WITH_VOLUME = "divergence_with_volume"
    COMPRESSION = "compression"
    EXPANSION = "expansion"


@dataclass(frozen=True, slots=True)
class IndicatorVariant:
    indicator: str
    parameters: Mapping[str, int]
    family: str
    source_id: str
    required_columns: tuple[str, ...]

    @property
    def token(self) -> str:
        return "_".join(f"{key}{value}" for key, value in sorted(self.parameters.items()))


_TRANSFORMATIONS = tuple(TechnicalTransformation)
_OSCILLATORS = frozenset({"RSI", "KDJ", "CMO", "RVI", "PSY", "VR"})


def technical_factor_library_v2() -> tuple[AtomicFactor, ...]:
    """Build a controlled, materializable L2 library without an unbounded Cartesian search."""
    factors = tuple(
        _build_factor(variant, transformation)
        for variant in _indicator_variants()
        for transformation in _TRANSFORMATIONS
    )
    ids = {factor.spec.factor_id for factor in factors}
    hashes = {factor.spec.expression_hash for factor in factors}
    if len(ids) != len(factors) or len(hashes) != len(factors):
        raise RuntimeError("technical v2 generation produced duplicate ids or expressions")
    return factors


def _indicator_variants() -> tuple[IndicatorVariant, ...]:
    variants: list[IndicatorVariant] = []

    def add(
        indicator: str,
        family: str,
        source_id: str,
        columns: tuple[str, ...],
        parameters: tuple[dict[str, int], ...],
    ) -> None:
        variants.extend(
            IndicatorVariant(indicator, parameter, family, source_id, columns)
            for parameter in parameters
        )

    add(
        "SMA",
        "trend",
        "SRC_BROCK_LAKONISHOK_LEBARON",
        ("close",),
        _windows(5, 10, 20, 60, 120, 252),
    )
    add("EMA", "trend", "SRC_BOC_TECH_20230913", ("close",), _windows(5, 10, 20, 60, 120))
    add("WMA", "trend", "SRC_BOC_TECH_20230913", ("close",), _windows(5, 10, 20, 60))
    add(
        "MACD",
        "trend",
        "SRC_BOC_TECH_20230913",
        ("close",),
        (
            {"fast": 6, "slow": 13, "signal": 5},
            {"fast": 12, "slow": 26, "signal": 9},
            {"fast": 24, "slow": 52, "signal": 18},
        ),
    )
    add("RSI", "oscillator", "SRC_BOC_TECH_20230913", ("close",), _windows(6, 14, 28))
    add("BOLL", "channel", "SRC_BOC_TECH_20230913", ("close",), _windows(10, 20, 60))
    add("KDJ", "oscillator", "SRC_BOC_TECH_20230913", ("high", "low", "close"), _windows(9, 14, 21))
    add(
        "ATR",
        "volatility",
        "SRC_DONGHAI_TECH_ML_20201120",
        ("high", "low", "close"),
        _windows(10, 14, 20, 60),
    )
    add(
        "TR",
        "volatility",
        "SRC_DONGHAI_TECH_ML_20201120",
        ("high", "low", "close"),
        _windows(5, 14),
    )
    add(
        "ADX",
        "trend",
        "SRC_DONGHAI_TECH_ML_20201120",
        ("high", "low", "close"),
        _windows(10, 14, 20),
    )
    add(
        "DMI",
        "trend",
        "SRC_DONGHAI_TECH_ML_20201120",
        ("high", "low", "close"),
        _windows(10, 14, 20),
    )
    add(
        "CCI",
        "oscillator",
        "SRC_DONGHAI_TECH_ML_20201120",
        ("high", "low", "close"),
        _windows(10, 14, 20, 60),
    )
    add("ROC", "momentum", "SRC_CSC_TECH_HF_20240623", ("close",), _windows(5, 10, 20, 60, 120))
    add("MTM", "momentum", "SRC_CSC_TECH_HF_20240623", ("close",), _windows(5, 10, 20, 60, 120))
    add("CMO", "oscillator", "SRC_BOC_TECH_20230913", ("close",), _windows(6, 14, 28))
    add(
        "RVI",
        "oscillator",
        "SRC_BOC_TECH_20230913",
        ("open", "high", "low", "close"),
        _windows(10, 14, 20),
    )
    add("TRIX", "trend", "SRC_BOC_TECH_20230913", ("close",), _windows(9, 12, 24))
    add("BIAS", "reversal", "SRC_BOC_TECH_20230913", ("close",), _windows(5, 10, 20, 60, 120))
    add("OBV", "volume_price", "SRC_BOC_TECH_20230913", ("close", "volume"), _windows(5, 20, 60))
    add("VR", "volume_price", "SRC_BOC_TECH_20230913", ("close", "volume"), _windows(14, 26, 60))
    add("PSY", "oscillator", "SRC_BOC_TECH_20230913", ("close",), _windows(6, 12, 24))
    add("FORCE", "volume_price", "SRC_BOC_TECH_20230913", ("close", "volume"), _windows(2, 13, 26))
    add("VHF", "trend", "SRC_BOC_TECH_20230913", ("close",), _windows(14, 28, 60, 120))
    add("AMA", "trend", "SRC_BOC_TECH_20230913", ("close",), _windows(10, 20, 60))
    add("COPPOCK", "momentum", "SRC_CSC_TECH_HF_20240623", ("close",), _windows(10, 14, 20))
    add(
        "DONCHIAN",
        "channel",
        "SRC_DONGHAI_TECH_ML_20201120",
        ("high", "low", "close"),
        _windows(10, 20, 30, 60, 120),
    )
    add("HURST", "regime", "SRC_DONGHAI_TECH_ML_20201120", ("close",), _windows(30, 60, 120, 252))
    return tuple(variants)


def _windows(*windows: int) -> tuple[dict[str, int], ...]:
    return tuple({"window": window} for window in windows)


def _build_factor(
    variant: IndicatorVariant,
    transformation: TechnicalTransformation,
) -> AtomicFactor:
    factor_id = f"tech_{variant.indicator.lower()}_{variant.token}_{transformation.value}"
    parameters: dict[str, Any] = {
        **variant.parameters,
        "indicator": variant.indicator,
        "transformation": transformation.value,
        "parameter_space": "technical_v2_controlled",
        "factor_packs": _factor_packs(variant.family, transformation),
    }
    lookback = max(variant.parameters.values()) + _transformation_history(transformation)
    role = (
        FactorRole.STATE_FEATURE
        if transformation
        in {
            TechnicalTransformation.CROSS_STATE,
            TechnicalTransformation.STATE_DURATION,
            TechnicalTransformation.RECOVERY_FROM_EXTREME,
            TechnicalTransformation.COMPRESSION,
            TechnicalTransformation.EXPANSION,
        }
        else FactorRole.ALPHA_CANDIDATE
    )
    spec = FactorSpec(
        factor_id=factor_id,
        name=f"{variant.indicator} {variant.token} {transformation.value}",
        description=f"{transformation.value} representation of {variant.indicator}.",
        family=variant.family,
        subfamily=variant.indicator.lower(),
        role=role,
        layer=FactorLayer.L2B,
        version="2.0.0",
        status=FactorStatus.DRAFT,
        hypothesis=(
            "A controlled technical representation may carry complementary "
            "cross-sectional information."
        ),
        expected_direction=0,
        implementation=f"aquant.factors.atomic.technical_v2:{factor_id}",
        input_fields=variant.required_columns,
        required_datasets=("bars_1d",),
        parent_factor_ids=(),
        variant_dimension="technical_transformation",
        required_history=lookback,
        minimum_periods=max(1, max(variant.parameters.values())),
        data_lag=1,
        availability_lag=1,
        universe="all_a_share",
        target_horizons=(1, 5, 10, 20, 60),
        parameters=parameters,
        source_type=SourceType.BROKER_REPORT,
        source_reference=variant.source_id,
        source_id=variant.source_id,
        source_formula_id=variant.indicator,
        source_faithfulness=SourceFaithfulness.DERIVED_VARIANT,
        implementation_notes="Close-derived values are usable only from the next tradable session.",
        normalization=transformation.value.upper(),
        missing_policy="PRESERVE",
        warmup_policy="REQUIRE_MINIMUM_PERIODS",
        complexity_score=float(3 + len(parameters) + _transformation_history(transformation)),
        variant_of=f"technical_base:{variant.indicator}:{variant.token}",
        implementation_status=ImplementationStatus.IMPLEMENTED,
        tags=(variant.family, variant.indicator.lower(), "technical_v2", "pit_safe"),
    )

    def calculate(panel: FactorPanelInput) -> Array:
        base = _base_indicator(variant, panel)
        return _transform(base, panel, transformation, variant)

    return AtomicFactor(spec, calculate)


def _field(panel: FactorPanelInput, name: str) -> Array:
    return np.asarray(panel.fields[name], dtype=np.float64)


def _base_indicator(variant: IndicatorVariant, panel: FactorPanelInput) -> Array:
    name, p = variant.indicator, variant.parameters
    close = _field(panel, "close")
    window = p.get("window", p.get("slow", 20))
    if name == "SMA":
        return rolling_mean(close, window)
    if name == "EMA":
        return ema(close, window, min_periods=window)
    if name == "WMA":
        return wma(close, window)
    if name == "MACD":
        line = ema(close, p["fast"], min_periods=p["fast"]) - ema(
            close, p["slow"], min_periods=p["slow"]
        )
        return line - ema(line, p["signal"], min_periods=p["signal"])
    if name == "RSI":
        change = delta(close)
        gain, loss = np.maximum(change, 0), np.maximum(-change, 0)
        return 100 * safe_div(sma_cn(gain, window), sma_cn(gain + loss, window))
    if name == "BOLL":
        return safe_div(close - rolling_mean(close, window), rolling_std(close, window, ddof=0))
    if name == "KDJ":
        low, high = _field(panel, "low"), _field(panel, "high")
        rsv = 100 * safe_div(
            close - rolling_min(low, window), rolling_max(high, window) - rolling_min(low, window)
        )
        k = sma_cn(rsv, 3)
        return 3 * k - 2 * sma_cn(k, 3)
    if name in {"TR", "ATR"}:
        true_range = _true_range(panel)
        return true_range if name == "TR" else ema(true_range, 2 * window - 1, min_periods=window)
    if name in {"ADX", "DMI"}:
        return _directional_indicator(panel, window, adx=name == "ADX")
    if name == "CCI":
        typical = (_field(panel, "high") + _field(panel, "low") + close) / 3
        mean = rolling_mean(typical, window)
        deviation = rolling_mean(np.abs(typical - mean), window)
        return safe_div(typical - mean, 0.015 * deviation)
    if name == "ROC":
        return 100 * (safe_div(close, delay(close, window)) - 1)
    if name == "MTM":
        return delta(close, window)
    if name == "CMO":
        change = delta(close)
        up = rolling_sum(np.maximum(change, 0), window)
        down = rolling_sum(np.maximum(-change, 0), window)
        return 100 * safe_div(up - down, up + down)
    if name == "RVI":
        return 100 * safe_div(
            rolling_mean(close - _field(panel, "open"), window),
            rolling_mean(_field(panel, "high") - _field(panel, "low"), window),
        )
    if name == "TRIX":
        first = ema(close, window, min_periods=window)
        second = ema(first, window, min_periods=window)
        third = ema(second, window, min_periods=window)
        return 100 * safe_div(delta(third), delay(third))
    if name == "BIAS":
        average = rolling_mean(close, window)
        return 100 * safe_div(close - average, average)
    if name == "OBV":
        signed_volume = np.sign(delta(close)) * _field(panel, "volume")
        return rolling_sum(signed_volume, window)
    if name == "VR":
        change, volume = delta(close), _field(panel, "volume")
        up = rolling_sum(np.where(change > 0, volume, 0), window)
        down = rolling_sum(np.where(change < 0, volume, 0), window)
        flat = rolling_sum(np.where(change == 0, volume, 0), window)
        return 100 * safe_div(up + 0.5 * flat, down + 0.5 * flat)
    if name == "PSY":
        return 100 * safe_div(rolling_sum(delta(close) > 0, window), window)
    if name == "FORCE":
        return ema(delta(close) * _field(panel, "volume"), window, min_periods=window)
    if name == "VHF":
        return safe_div(
            rolling_max(close, window) - rolling_min(close, window),
            rolling_sum(np.abs(delta(close)), window),
        )
    if name == "AMA":
        efficiency = safe_div(
            np.abs(delta(close, window)), rolling_sum(np.abs(delta(close)), window)
        )
        smoothing = np.square(efficiency * (2 / 3 - 2 / 31) + 2 / 31)
        return _adaptive_average(close, smoothing)
    if name == "COPPOCK":
        raw = 100 * (safe_div(close, delay(close, window + 3)) - 1)
        raw += 100 * (safe_div(close, delay(close, window + 1)) - 1)
        return wma(raw, window)
    if name == "DONCHIAN":
        high = rolling_max(_field(panel, "high"), window)
        low = rolling_min(_field(panel, "low"), window)
        return safe_div(close - low, high - low)
    if name == "HURST":
        return _hurst_proxy(close, window)
    raise ValueError(f"unsupported technical indicator: {name}")


def _true_range(panel: FactorPanelInput) -> Array:
    high, low, previous = _field(panel, "high"), _field(panel, "low"), delay(_field(panel, "close"))
    return np.asarray(
        np.maximum.reduce((high - low, np.abs(high - previous), np.abs(low - previous))),
        dtype=np.float64,
    )


def _directional_indicator(panel: FactorPanelInput, window: int, *, adx: bool) -> Array:
    high, low = _field(panel, "high"), _field(panel, "low")
    up, down = delta(high), -delta(low)
    plus = np.where((up > down) & (up > 0), up, 0)
    minus = np.where((down > up) & (down > 0), down, 0)
    atr = rolling_sum(_true_range(panel), window)
    plus_di = 100 * safe_div(rolling_sum(plus, window), atr)
    minus_di = 100 * safe_div(rolling_sum(minus, window), atr)
    dx = 100 * safe_div(np.abs(plus_di - minus_di), plus_di + minus_di)
    return rolling_mean(dx, window) if adx else plus_di - minus_di


def _adaptive_average(values: Array, smoothing: Array) -> Array:
    result = np.full(values.shape, np.nan)
    for column in range(values.shape[1]):
        state = np.nan
        for row in range(values.shape[0]):
            value, weight = values[row, column], smoothing[row, column]
            if not np.isfinite(value) or not np.isfinite(weight):
                continue
            state = value if not np.isfinite(state) else state + weight * (value - state)
            result[row, column] = state
    return result


def _hurst_proxy(values: Array, window: int) -> Array:
    half = max(2, window // 2)
    short_scale = rolling_std(delta(values), half, ddof=0)
    long_scale = rolling_std(delta(values), window, ddof=0)
    with np.errstate(all="ignore"):
        ratio_log = np.log(safe_div(long_scale, short_scale))
    return safe_div(ratio_log, np.log(window / half)) + 0.5


def _transform(
    base: Array,
    panel: FactorPanelInput,
    transformation: TechnicalTransformation,
    variant: IndicatorVariant,
) -> Array:
    window = max(variant.parameters.values())
    short = max(3, min(10, window // 3))
    if transformation is TechnicalTransformation.LEVEL:
        return base
    if transformation is TechnicalTransformation.NORMALIZED_LEVEL:
        return safe_div(base, rolling_std(base, window, min_periods=max(2, window // 2), ddof=0))
    if transformation is TechnicalTransformation.DELTA:
        return delta(base)
    if transformation is TechnicalTransformation.SLOPE:
        return regression_slope(base, short)
    if transformation is TechnicalTransformation.ACCELERATION:
        return delta(regression_slope(base, short))
    if transformation is TechnicalTransformation.TS_PERCENTILE:
        return rolling_rank(base, max(20, window), min_periods=window)
    if transformation is TechnicalTransformation.ZSCORE:
        return ts_zscore(base, max(20, window), min_periods=window)
    centered = base - _neutral_level(variant.indicator)
    if transformation is TechnicalTransformation.THRESHOLD_DISTANCE:
        return centered
    if transformation is TechnicalTransformation.CROSS_STATE:
        return np.sign(centered)
    if transformation is TechnicalTransformation.STATE_DURATION:
        return _state_duration(centered)
    if transformation is TechnicalTransformation.RECOVERY_FROM_EXTREME:
        percentile = rolling_rank(base, max(20, window), min_periods=window)
        return np.where((delay(percentile) < 0.2) & (percentile >= 0.2), percentile - 0.2, 0)
    if transformation is TechnicalTransformation.DIVERGENCE_WITH_PRICE:
        return -rolling_corr(delta(base), delta(_field(panel, "close")), window)
    if transformation is TechnicalTransformation.DIVERGENCE_WITH_VOLUME:
        return -rolling_corr(delta(base), delta(_field(panel, "volume")), window)
    ratio = safe_div(
        rolling_std(base, short, ddof=0),
        rolling_std(base, max(20, window), ddof=0),
    )
    if transformation is TechnicalTransformation.COMPRESSION:
        return -ratio
    if transformation is TechnicalTransformation.EXPANSION:
        return ratio
    raise ValueError(f"unsupported transformation: {transformation}")


def _neutral_level(indicator: str) -> float:
    if indicator in {"RSI", "KDJ", "PSY"}:
        return 50.0
    if indicator == "VR":
        return 100.0
    if indicator == "DONCHIAN":
        return 0.5
    return 0.0


def _state_duration(centered: Array) -> Array:
    result = np.full(centered.shape, np.nan)
    for column in range(centered.shape[1]):
        duration = 0
        prior = 0.0
        for row, value in enumerate(centered[:, column]):
            if not np.isfinite(value):
                duration, prior = 0, 0.0
                continue
            state = float(np.sign(value))
            duration = duration + 1 if state != 0 and state == prior else int(state != 0)
            result[row, column] = state * duration
            prior = state
    return result


def _transformation_history(transformation: TechnicalTransformation) -> int:
    if transformation in {
        TechnicalTransformation.LEVEL,
        TechnicalTransformation.THRESHOLD_DISTANCE,
        TechnicalTransformation.CROSS_STATE,
        TechnicalTransformation.STATE_DURATION,
    }:
        return 1
    if transformation in {TechnicalTransformation.DELTA, TechnicalTransformation.ACCELERATION}:
        return 2
    return 20


def _factor_packs(
    family: str,
    transformation: TechnicalTransformation,
) -> tuple[str, str]:
    if transformation is TechnicalTransformation.LEVEL:
        representation = "technical_classic_level_v1"
    elif transformation in {
        TechnicalTransformation.DELTA,
        TechnicalTransformation.SLOPE,
        TechnicalTransformation.ACCELERATION,
    }:
        representation = "technical_classic_change_v1"
    elif transformation in {
        TechnicalTransformation.DIVERGENCE_WITH_PRICE,
        TechnicalTransformation.DIVERGENCE_WITH_VOLUME,
    }:
        representation = "technical_classic_divergence_v1"
    else:
        representation = "technical_classic_state_v1"
    canonical_family = {
        "trend": "technical_trend_v1",
        "momentum": "technical_momentum_v1",
        "reversal": "technical_reversal_v1",
        "oscillator": "technical_oscillator_v1",
        "channel": "technical_channel_v1",
        "volatility": "technical_volatility_v1",
        "volume_price": "technical_volume_price_v1",
        "regime": "technical_regime_interaction_v1",
    }[family]
    return representation, canonical_family
