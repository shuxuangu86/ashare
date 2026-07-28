from datetime import date, timedelta

import numpy as np
import pytest

from aquant.factors.preprocessing import (
    PITExposurePanel,
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
