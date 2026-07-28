import numpy as np

from aquant.factors.selection import converge_factors, cross_sectional_spearman


def test_three_correlation_views_cluster_and_preserve_conditional_information() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(80, 30))
    values = {
        "a": base,
        "b": base + rng.normal(scale=0.02, size=base.shape),
        "c": rng.normal(size=base.shape),
    }
    forward = base * 0.01 + rng.normal(scale=0.01, size=base.shape)
    rank_ic = {
        "a": np.linspace(0.01, 0.04, 80),
        "b": np.linspace(0.011, 0.041, 80),
        "c": rng.normal(scale=0.02, size=80),
    }
    long_short = {
        "a": np.linspace(0.001, 0.004, 80),
        "b": np.linspace(0.0011, 0.0041, 80),
        "c": rng.normal(scale=0.002, size=80),
    }
    result = converge_factors(
        values,
        forward,
        long_short,
        rank_ic,
        {"a": 0.7, "b": 0.8, "c": 0.5},
        maximum_distance=0.25,
    )
    by_id = {item.factor_id: item for item in result.factors}
    assert by_id["a"].cluster_id == by_id["b"].cluster_id
    assert by_id["a"].representative_factor_id == "b"
    assert by_id["b"].is_representative
    assert not by_id["c"].is_representative or by_id["c"].representative_factor_id == "c"
    assert result.value_spearman.shape == (3, 3)
    assert result.long_short_correlation.shape == (3, 3)
    assert result.rank_ic_correlation.shape == (3, 3)
    assert np.isfinite(by_id["a"].conditional_rank_ic)


def test_cross_sectional_spearman_is_tie_safe_and_deterministic() -> None:
    values = {
        "a": np.array([[1, 1, 2, 3], [2, 3, 4, 5]], dtype=float),
        "b": np.array([[2, 2, 4, 6], [4, 6, 8, 10]], dtype=float),
    }
    first = cross_sectional_spearman(values)
    second = cross_sectional_spearman(values)
    np.testing.assert_array_equal(first, second)
    assert first[0, 1] == 1


def test_cross_sectional_spearman_handles_pairwise_missing_values() -> None:
    values = {
        "a": np.array([[1, 2, 3, np.nan], [1, 2, 3, 4]], dtype=float),
        "b": np.array([[2, 4, 6, 8], [4, 3, 2, 1]], dtype=float),
        "c": np.full((2, 4), np.nan),
    }
    result = cross_sectional_spearman(values)
    # Each factor is ranked on its own available cross-section before pairwise overlap.
    assert -0.08 < result[0, 1] < -0.07
    assert result[0, 2] == 0
    assert np.all(np.diag(result) == 1)
