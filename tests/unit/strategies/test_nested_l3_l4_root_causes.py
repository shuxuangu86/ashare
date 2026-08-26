import numpy as np
import pytest
from scripts.audit_nested_l3_l4_root_causes import (
    _benjamini_hochberg,
    _block_bootstrap_mean,
    _cross_section_metrics,
    _pit_labels,
    _rank_correlation,
)


def test_pit_labels_match_nested_t_plus_one_horizon() -> None:
    prices = np.asarray([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [8.0]])

    labels = _pit_labels(prices, 5)

    assert labels[0, 0] == pytest.approx(3.0)
    assert np.isnan(labels[1:, 0]).all()


def test_cross_section_metrics_separate_global_rank_from_top_tail() -> None:
    score = np.arange(100, dtype=float)
    returns = score.copy()
    returns[-20:] = returns[-20:][::-1]

    metrics, deciles = _cross_section_metrics(score, returns, target_count=20)

    assert metrics["rank_ic"] > 0
    assert metrics["tail_rank_ic"] < 0
    assert len(deciles) == 10


def test_block_bootstrap_is_deterministic_and_detects_positive_mean() -> None:
    values = np.linspace(0.01, 0.02, 100)

    first = _block_bootstrap_mean(values, block_size=10, replicates=200, seed=7)
    second = _block_bootstrap_mean(values, block_size=10, replicates=200, seed=7)

    assert first == second
    assert first[0] > 0
    assert first[2] == 0


def test_benjamini_hochberg_is_monotone_in_original_order() -> None:
    adjusted = _benjamini_hochberg([0.01, 0.04, 0.03, 0.20])

    assert adjusted == pytest.approx([0.04, 0.0533333333, 0.0533333333, 0.20])


def test_rank_correlation_is_invariant_to_tie_permutation() -> None:
    left = np.asarray([1.0, 1.0, 2.0, 2.0])
    right = np.asarray([1.0, 2.0, 3.0, 4.0])
    permutation = np.asarray([3, 0, 2, 1])

    assert _rank_correlation(left, right) == pytest.approx(
        _rank_correlation(left[permutation], right[permutation])
    )
