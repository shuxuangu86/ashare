from datetime import date, timedelta
from pathlib import Path

import numpy as np

from aquant.factors.atomic import technical_factor_library_v2
from aquant.factors.atomic.models import FactorPanelInput
from aquant.factors.evaluation.eligibility import (
    EligibilityReason,
    FeatureEvidence,
    ResearchSafetyEvidence,
    evaluate_feature_eligibility,
    evaluate_research_safety,
    family_diversity_selection,
)
from aquant.factors.operators.time_series import sma_cn, wma
from aquant.factors.sources import SourceRegistry
from aquant.factors.spec import FactorStatus


def test_chinese_sma_and_wma_golden_sequences() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(sma_cn(values, 3, 1), [1, 4 / 3, 17 / 9, 70 / 27])
    np.testing.assert_allclose(wma(values, 3), [np.nan, np.nan, 14 / 6, 20 / 6], equal_nan=True)


def test_technical_v2_is_controlled_unique_and_traceable() -> None:
    factors = technical_factor_library_v2()
    assert 1_000 <= len(factors) < 3_000
    assert len({factor.spec.factor_id for factor in factors}) == len(factors)
    assert len({factor.spec.expression_hash for factor in factors}) == len(factors)
    assert all(factor.spec.availability_lag == 1 for factor in factors)
    assert all(factor.spec.source_id.startswith("SRC_") for factor in factors)
    assert all(factor.spec.minimum_periods for factor in factors)


def test_technical_v2_smoke_is_deterministic_and_has_no_future_dependency() -> None:
    rows, columns = 300, 2
    close = np.arange(rows * columns, dtype=float).reshape(rows, columns) / 10 + 10
    panel = FactorPanelInput(
        tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(rows)),
        ("000001.SZ", "600000.SH"),
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full_like(close, 1_000.0),
        },
    )
    factors = technical_factor_library_v2()
    sample = (
        next(item for item in factors if item.spec.factor_id == "tech_rsi_window14_level"),
        next(
            item
            for item in factors
            if item.spec.factor_id == "tech_macd_fast12_signal9_slow26_zscore"
        ),
        next(item for item in factors if item.spec.factor_id == "tech_donchian_window20_level"),
    )
    for factor in sample:
        first = factor.compute_array(panel)
        np.testing.assert_allclose(first, factor.compute_array(panel), equal_nan=True)
        changed_close = close.copy()
        changed_close[-1] *= 10
        changed = FactorPanelInput(
            panel.trade_dates,
            panel.ts_codes,
            {**panel.fields, "close": changed_close},
        )
        second = factor.compute_array(changed)
        np.testing.assert_allclose(first[:-1], second[:-1], equal_nan=True)

    representatives = {}
    for factor in factors:
        indicator = str(factor.spec.parameters["indicator"])
        transformation = str(factor.spec.parameters["transformation"])
        if transformation == "level":
            representatives.setdefault(f"indicator:{indicator}", factor)
        if indicator == "RSI":
            representatives.setdefault(f"transformation:{transformation}", factor)
    assert len(representatives) == 42
    for factor in representatives.values():
        values = factor.compute_array(panel)
        assert values.shape == close.shape
        assert not np.any(np.isinf(values))


def test_source_registry_is_unique_and_stable() -> None:
    path = Path("configs/factors/source_registry.yaml")
    first = SourceRegistry.from_yaml(path)
    second = SourceRegistry.from_yaml(path)
    assert len(first.sources) >= 10
    assert first.content_hash == second.content_hash
    assert first.get("SRC_ALPHA101").identifier == "arXiv:1601.00991"


def test_research_and_feature_gates_are_separate_from_production() -> None:
    research = evaluate_research_safety(ResearchSafetyEvidence(True, True, True, True, True, 0.9))
    assert research.tier is FactorStatus.RESEARCH_VALIDATED
    feature = evaluate_feature_eligibility(
        FeatureEvidence(0.005, 0.2, 0.3, 0.1, 0.9, 0.5, 0.2, 4, 0.8, 0.4)
    )
    assert feature.tier is FactorStatus.FEATURE_ELIGIBLE
    assert feature.reason_codes == (EligibilityReason.PASSED_FEATURE_ELIGIBILITY,)


def test_family_selection_keeps_required_archetypes() -> None:
    evidence = {
        f"factor_{index}": FeatureEvidence(
            0.005 + index / 10_000,
            0.2 + index / 100,
            0.1 + index / 50,
            0.2,
            0.9,
            1 / (index + 1),
            0.1,
            float(index + 1),
            0.9 - index / 20,
            index / 20,
        )
        for index in range(8)
    }
    selected = family_diversity_selection(evidence)
    reasons = {reason for factor_reasons in selected.values() for reason in factor_reasons}
    assert EligibilityReason.FAMILY_STRONGEST in reasons
    assert EligibilityReason.FAMILY_MOST_STABLE in reasons
    assert EligibilityReason.FAMILY_LOWEST_TURNOVER in reasons
    assert EligibilityReason.FAMILY_MOST_INDEPENDENT in reasons
    assert EligibilityReason.FAMILY_REGIME_COMPLEMENT in reasons
