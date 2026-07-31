import re
from datetime import date, timedelta

import numpy as np

from aquant.factors.atomic import alpha101_original_library, gtja191_original_library
from aquant.factors.atomic.models import FactorPanelInput
from aquant.factors.atomic.published_formulas import normalize_published_formula
from aquant.factors.spec import ImplementationStatus, SourceFaithfulness


def _panel(*, future_multiplier: float = 1.0, cutoff: int = 300) -> FactorPanelInput:
    rows, columns = 320, 20
    generator = np.random.default_rng(20260731)
    innovations = generator.normal(0.0004, 0.015, size=(rows, columns))
    close = np.exp(np.cumsum(innovations, axis=0)) * (10 + np.arange(columns, dtype=float)[None, :])
    open_ = close * (1 + generator.normal(0, 0.004, size=(rows, columns)))
    spread = generator.uniform(0.005, 0.025, size=(rows, columns))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = np.maximum(
        1_000_000 + generator.normal(0, 100_000, size=(rows, columns)),
        10_000,
    )
    fields = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": volume * (open_ + high + low + close) / 4,
        "total_market_cap": close
        * (1_000_000_000 + np.arange(columns, dtype=float)[None, :] * 10_000_000),
    }
    if future_multiplier != 1:
        for values in fields.values():
            values[cutoff:] *= future_multiplier
    start = date(2020, 1, 1)
    return FactorPanelInput(
        tuple(start + timedelta(days=index) for index in range(rows)),
        tuple(f"{index:06d}.SZ" for index in range(columns)),
        fields,
    )


def test_published_formula_catalogs_are_complete_and_traceable() -> None:
    alpha101 = alpha101_original_library()
    alpha191 = gtja191_original_library()
    assert len(alpha101) == 101
    assert len(alpha191) == 191
    assert len({factor.spec.factor_id for factor in (*alpha101, *alpha191)}) == 292
    assert all(factor.spec.availability_lag == 1 for factor in (*alpha101, *alpha191))
    assert all(factor.spec.source_page for factor in (*alpha101, *alpha191))
    assert all(
        len(str(factor.spec.parameters["source_pdf_sha256"])) == 64
        for factor in (*alpha101, *alpha191)
    )


def test_published_formula_faithfulness_and_ambiguity_are_explicit() -> None:
    alpha101 = alpha101_original_library()
    alpha191 = gtja191_original_library()
    assert all(
        factor.spec.implementation_status is ImplementationStatus.IMPLEMENTED for factor in alpha101
    )
    ambiguous = [
        factor
        for factor in alpha191
        if factor.spec.implementation_status is ImplementationStatus.FORMULA_AMBIGUOUS
    ]
    assert [factor.spec.factor_id for factor in ambiguous] == ["gtja191_143"]
    assert any(
        factor.spec.source_faithfulness is SourceFaithfulness.CORRECTED_AMBIGUITY
        for factor in alpha191
    )
    assert all(
        "raw_formula" in factor.spec.parameters and "adopted_formula" in factor.spec.parameters
        for factor in alpha191
    )


def test_gtja_derived_names_declare_their_underlying_input_fields() -> None:
    alpha191 = gtja191_original_library()
    assert set(alpha191[68].spec.input_fields) >= {"open", "high", "low"}
    tr_factors = [
        factor
        for factor in alpha191
        if re.search(r"\bTR\b", str(factor.spec.parameters["adopted_formula"]), re.IGNORECASE)
    ]
    assert tr_factors
    assert all(set(factor.spec.input_fields) >= {"high", "low", "close"} for factor in tr_factors)


def test_all_executable_published_formulas_are_deterministic() -> None:
    panel = _panel()
    factors = (*alpha101_original_library(), *gtja191_original_library())
    for factor in factors:
        first = factor.compute_array(panel)
        second = factor.compute_array(panel)
        np.testing.assert_allclose(first, second, equal_nan=True)
        if factor.spec.implementation_status is ImplementationStatus.IMPLEMENTED:
            assert np.count_nonzero(np.isfinite(first)) > 0, factor.spec.factor_id


def test_representative_published_formulas_have_no_future_dependency() -> None:
    cutoff = 300
    original = _panel(cutoff=cutoff)
    changed = _panel(future_multiplier=1000, cutoff=cutoff)
    factors = (
        alpha101_original_library()[0],
        alpha101_original_library()[95],
        gtja191_original_library()[0],
        gtja191_original_library()[179],
    )
    for factor in factors:
        np.testing.assert_allclose(
            factor.compute_array(original)[:cutoff],
            factor.compute_array(changed)[:cutoff],
            equal_nan=True,
            err_msg=factor.spec.factor_id,
        )


def test_gtja_nested_ternary_normalization_preserves_function_window() -> None:
    normalized = normalize_published_formula(
        "SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)"
    )
    assert normalized == (
        "SUM((where((CLOSE>DELAY(CLOSE,1)), (VOLUME), "
        "((where((CLOSE<DELAY(CLOSE,1)), (-VOLUME), (0)))))),20)"
    )
