import hashlib
from datetime import date

import numpy as np
import pytest

from aquant.domain.data_release import DataReleaseId
from aquant.factors import FactorCacheKey, FactorValueCache, classic_factor_library
from aquant.factors.preprocessing import (
    neutralize,
    preprocess_cross_section,
    winsorize_mad,
    zscore,
)


def test_winsorize_zscore_and_missing_values() -> None:
    values = np.array([1.0, 2.0, 3.0, 1000.0, np.nan])
    clipped = winsorize_mad(values, scale=3)
    standardized = zscore(clipped)
    assert clipped[3] < 1000
    assert np.isnan(standardized[-1])
    assert np.isclose(np.nanmean(standardized), 0.0)
    assert np.isclose(np.nanstd(standardized), 1.0)


def test_neutralization_removes_linear_exposure() -> None:
    exposure = np.array([[1.0], [2.0], [3.0], [4.0]])
    values = np.array([3.0, 5.0, 7.0, 9.0])
    residual = neutralize(values, exposure)
    assert np.allclose(residual, 0.0, atol=1e-12)
    assert np.allclose(preprocess_cross_section(values, exposures=exposure), 0.0)


def test_preprocessing_validates_parameters_and_shapes() -> None:
    with pytest.raises(ValueError, match="positive"):
        winsorize_mad(np.array([1.0]), scale=0)
    with pytest.raises(ValueError, match="requires"):
        neutralize(np.array([1.0]), np.array([1.0]))
    assert np.isnan(zscore(np.array([np.nan]))).all()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_factor_cache_is_lineage_bound_copy_safe_and_incremental() -> None:
    release = DataReleaseId("cn_equity_20260716_001")
    key = FactorCacheKey(_hash("factor"), date(2026, 7, 15), release, _hash("params"))
    cache = FactorValueCache()
    values = np.array([1.0, np.nan])
    cache.put(key, values)
    values[0] = 9

    restored = cache.get(key)
    assert restored is not None and restored[0] == 1
    restored[0] = 8
    assert cache.get(key)[0] == 1  # type: ignore[index]
    assert cache.missing_dates(
        expression_hash=key.expression_hash,
        dates=(date(2026, 7, 15), date(2026, 7, 16)),
        data_release_id=release,
        parameter_hash=key.parameter_hash,
    ) == (date(2026, 7, 16),)


def test_factor_cache_refuses_conflicting_result_and_bad_hash() -> None:
    release = DataReleaseId("cn_equity_20260716_001")
    key = FactorCacheKey(_hash("factor"), date(2026, 7, 16), release, _hash("params"))
    cache = FactorValueCache()
    cache.put(key, np.array([1.0]))
    with pytest.raises(ValueError, match="different"):
        cache.put(key, np.array([2.0]))
    with pytest.raises(ValueError, match="SHA-256"):
        FactorCacheKey("bad", date(2026, 7, 16), release, _hash("params"))


def test_classic_factor_library_is_versioned_and_deduplicated() -> None:
    factors = classic_factor_library()
    assert {factor.name for factor in factors} == {
        "momentum_20d",
        "reversal_5d",
        "volatility_20d",
        "turnover_proxy_20d",
    }
    assert len({factor.expression.expression_hash for factor in factors}) == len(factors)
    assert all(factor.version == "1.0.0" and factor.hypothesis for factor in factors)
