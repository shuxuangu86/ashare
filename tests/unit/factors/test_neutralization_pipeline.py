from datetime import date, timedelta

import numpy as np
import pytest

from aquant.data.industry.models import IndustryPITPanel
from aquant.factors.atomic.models import FactorPanelInput
from aquant.factors.preprocessing import (
    PITExposurePanel,
    load_pit_exposures,
    load_size_exposures,
    neutralization_variants,
    neutralize_pit_factor,
)


def test_raw_industry_and_industry_size_variants_are_separate_and_pit_safe() -> None:
    rng = np.random.default_rng(11)
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(5))
    groups = np.array([["bank"] * 20 + ["tech"] * 20] * len(dates), dtype=object)
    size = np.broadcast_to(np.linspace(1e9, 5e10, 40), (len(dates), 40)).copy()
    available = np.array([[value] * 40 for value in dates], dtype=object)
    values = np.log(size) + (groups == "tech") * 2 + rng.normal(scale=0.1, size=size.shape)
    panel = PITExposurePanel(dates, groups, size, available)
    variants = neutralization_variants(values, panel, minimum_observations=20)
    assert set(variants) == {
        "raw",
        "industry_neutral",
        "size_neutral",
        "industry_size_neutral",
    }
    assert not np.shares_memory(variants["raw"], values)
    for row in variants["industry_size_neutral"]:
        valid = np.isfinite(row)
        log_size = np.log(size[0, valid])
        weights = np.sqrt(size[0, valid])
        weighted_inner = np.sum(weights * row[valid] * log_size)
        scale = np.linalg.norm(np.sqrt(weights) * row[valid]) * np.linalg.norm(
            np.sqrt(weights) * log_size
        )
        assert abs(weighted_inner / scale) < 1e-10


def test_future_or_misaligned_exposures_are_rejected() -> None:
    dates = (date(2020, 1, 1),)
    groups = np.array([["bank", "tech"]], dtype=object)
    cap = np.ones((1, 2))
    future = np.array([[date(2020, 1, 2)] * 2], dtype=object)
    with pytest.raises(ValueError, match="not available"):
        PITExposurePanel(dates, groups, cap, future)


def test_pit_neutralization_reports_exclusions_and_diagnostics() -> None:
    dates = (date(2024, 1, 2),)
    groups = np.array([["bank"] * 10 + ["tech"] * 10 + ["single"] + [None]], dtype=object)
    cap = np.linspace(1e9, 3e10, 22)[None, :]
    available = np.array([[dates[0]] * 22], dtype=object)
    values = np.log(cap) + (groups == "tech") * 2.0
    result = neutralize_pit_factor(
        values,
        PITExposurePanel(dates, groups, cap, available),
        minimum_observations=15,
        minimum_industry_observations=3,
    )
    assert result.diagnostics[0].status == "PASS"
    assert result.diagnostics[0].industries == 2
    assert result.neutralization_status[0, 20] == "SMALL_INDUSTRY"
    assert result.neutralization_status[0, 21] == "UNKNOWN_INDUSTRY"
    assert np.all(np.isfinite(result.neutralized_value[0, :20]))
    assert np.all(np.isnan(result.neutralized_value[0, 20:]))


def test_single_industry_cross_section_is_blocked() -> None:
    dates = (date(2024, 1, 2),)
    groups = np.array([["bank"] * 30], dtype=object)
    cap = np.linspace(1e9, 3e10, 30)[None, :]
    available = np.array([[dates[0]] * 30], dtype=object)
    result = neutralize_pit_factor(
        np.arange(30, dtype=float)[None, :],
        PITExposurePanel(dates, groups, cap, available),
    )
    assert result.diagnostics[0].status == "BLOCKED"
    assert np.all(np.isnan(result.neutralized_value))


def test_exposure_loaders_preserve_panel_alignment_and_availability() -> None:
    dates = (date(2024, 1, 2), date(2024, 1, 3))
    codes = ("000001.SZ", "600000.SH")
    cap = np.array([[1e9, 2e9], [1.1e9, 2.1e9]])
    panel = FactorPanelInput(dates, codes, {"float_market_cap": cap})
    industries = np.array([["bank", "bank"], ["bank", "bank"]], dtype=object)
    available = np.array([[dates[0]] * 2, [dates[1]] * 2], dtype=object)

    class RepositoryStub:
        def panel(
            self,
            trade_dates: tuple[date, ...],
            ts_codes: tuple[str, ...],
            *,
            level: str,
        ) -> IndustryPITPanel:
            assert (trade_dates, ts_codes, level) == (dates, codes, "L1")
            return IndustryPITPanel(dates, codes, industries, available)

    pit = load_pit_exposures(panel, RepositoryStub())  # type: ignore[arg-type]
    size = load_size_exposures(panel)
    assert np.array_equal(pit.industries, industries)
    assert np.array_equal(pit.float_market_cap, cap)
    assert all(value is None for value in size.industries.ravel())
    assert size.available_dates.tolist() == [[dates[0]] * 2, [dates[1]] * 2]


def test_exposure_loaders_and_variants_reject_invalid_contracts() -> None:
    panel = FactorPanelInput((date(2024, 1, 2),), ("000001.SZ",), {"close": np.ones((1, 1))})
    repository = object()
    with pytest.raises(KeyError, match="float_market_cap"):
        load_pit_exposures(panel, repository)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="float_market_cap"):
        load_size_exposures(panel)
    with pytest.raises(ValueError, match="fields must align"):
        PITExposurePanel(
            panel.trade_dates,
            np.empty((1, 2), dtype=object),
            np.ones((1, 1)),
            np.empty((1, 1), dtype=object),
        )
    valid = PITExposurePanel(
        panel.trade_dates,
        np.array([["bank"]], dtype=object),
        np.ones((1, 1)),
        np.array([[panel.trade_dates[0]]], dtype=object),
    )
    with pytest.raises(ValueError, match="panels must align"):
        neutralization_variants(np.ones((2, 1)), valid)
    with pytest.raises(ValueError, match="positive"):
        neutralization_variants(
            np.ones((1, 1)),
            valid,
            minimum_industry_observations=0,
        )


def test_size_only_neutralization_marks_missing_cap_without_industry() -> None:
    dates = (date(2024, 1, 2),)
    cap = np.linspace(1e9, 3e10, 30)[None, :]
    cap[0, 0] = np.nan
    exposures = PITExposurePanel(
        dates,
        np.full((1, 30), None, dtype=object),
        cap,
        np.array([[dates[0]] * 30], dtype=object),
    )
    result = neutralize_pit_factor(
        np.arange(30, dtype=float)[None, :],
        exposures,
        method="size",
    )
    assert result.neutralization_status[0, 0] == "MISSING_MARKET_CAP"
    assert np.count_nonzero(result.neutralization_status == "VALID") == 29


def test_industry_proxy_preserves_unknown_industries_and_removes_style_exposure() -> None:
    dates = (date(2024, 1, 2), date(2024, 1, 3))
    groups = np.array(
        [["bank"] * 15 + ["tech"] * 15 + [None] * 10] * len(dates),
        dtype=object,
    )
    cap = np.broadcast_to(np.linspace(1e9, 5e10, 40), (len(dates), 40)).copy()
    available = np.array([[value] * 40 for value in dates], dtype=object)
    size = np.log(cap)
    beta = np.broadcast_to(np.linspace(-1, 1, 40), cap.shape).copy()
    beta[0, 35] = np.nan
    values = size + 2 * (groups == "tech") + 0.5 * np.nan_to_num(beta)
    result = neutralize_pit_factor(
        values,
        PITExposurePanel(dates, groups, cap, available),
        method="industry_hybrid",
        style_exposures={"beta": beta, "size": size},
    )
    assert all(item.status == "HYBRID_PASS" for item in result.diagnostics)
    assert result.diagnostics[0].industry_coverage_ratio == pytest.approx(0.75)
    assert result.diagnostics[0].neutralization_basis == "PIT_INDUSTRY_PLUS_STYLE_FALLBACK"
    assert np.all(np.isfinite(result.neutralized_value))
    assert np.all(result.neutralization_status == "VALID_HYBRID")
    assert abs(result.diagnostics[0].size_correlation_after or 0) < 1e-10


def test_industry_proxy_requires_aligned_style_exposures() -> None:
    dates = (date(2024, 1, 2),)
    exposures = PITExposurePanel(
        dates,
        np.array([["bank"] * 20 + [None] * 10], dtype=object),
        np.linspace(1e9, 3e10, 30)[None, :],
        np.array([[dates[0]] * 30], dtype=object),
    )
    with pytest.raises(ValueError, match="requires PIT style"):
        neutralize_pit_factor(
            np.arange(30, dtype=float)[None, :],
            exposures,
            method="industry_proxy",
        )
    with pytest.raises(ValueError, match="must align"):
        neutralize_pit_factor(
            np.arange(30, dtype=float)[None, :],
            exposures,
            method="industry_proxy",
            style_exposures={"beta": np.ones((2, 30))},
        )
