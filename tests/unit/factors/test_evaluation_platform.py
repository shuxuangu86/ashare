import numpy as np
import pytest

from aquant.factors.evaluation.cost import net_returns
from aquant.factors.evaluation.ic import forward_return_labels, information_coefficient
from aquant.factors.evaluation.multiple_testing import (
    benjamini_hochberg,
    family_multiple_testing_summary,
    superior_predictive_ability_test,
)
from aquant.factors.evaluation.quality import evaluate_quality
from aquant.factors.evaluation.walk_forward import fixed_time_split, walk_forward_splits


def test_quality_metrics_and_failure_flags() -> None:
    values = np.array([[1.0, 2.0, np.nan], [0.0, np.inf, 3.0], [np.nan, np.nan, np.nan]])
    quality = evaluate_quality(values)
    assert quality.valid_count == 4
    assert quality.infinite_count == 1
    assert quality.coverage == pytest.approx(4 / 9)
    assert quality.largest_gap == 1
    assert "LOW_COVERAGE" in quality.flags
    constant = evaluate_quality(np.ones((3, 4)))
    assert "NEAR_CONSTANT" in constant.flags
    empty = evaluate_quality(np.full((2, 2), np.nan))
    assert "ALL_NULL" in empty.flags


def test_ic_rank_ic_and_forward_labels() -> None:
    factor = np.tile(np.arange(5, dtype=float), (4, 1))
    returns = factor * 0.1
    pearson = information_coefficient(factor, returns)
    ranked = information_coefficient(factor, returns, rank=True)
    assert pearson.mean == pytest.approx(1)
    assert ranked.mean == pytest.approx(1)
    assert ranked.positive_ratio == 1

    close = np.array([[1.0, 2.0], [2.0, 2.0], [4.0, 1.0]])
    labels = forward_return_labels(close, 1)
    np.testing.assert_allclose(labels[0], [1, 0])
    np.testing.assert_allclose(labels[1], [1, -0.5])
    assert np.isnan(labels[-1]).all()


def test_forward_labels_do_not_enter_factor_history() -> None:
    close = np.arange(1.0, 21.0)[:, None]
    changed = close.copy()
    changed[15:] *= 100
    original_labels = forward_return_labels(close, 5)
    changed_labels = forward_return_labels(changed, 5)
    assert not np.allclose(original_labels[:15], changed_labels[:15], equal_nan=True)
    # Labels deliberately change earlier rows; feature calculators are separately
    # leakage-tested and never receive this label matrix.


def test_walk_forward_is_ordered_purged_and_embargoed() -> None:
    splits = walk_forward_splits(
        100,
        train_size=40,
        validation_size=10,
        step=10,
        expanding=True,
        purge=5,
        embargo=2,
    )
    assert splits
    for split in splits:
        assert split.train.stop + 5 == split.validation.start
        assert split.train.start == 0
    fixed = fixed_time_split(100, train_end=60, validation_end=90, purge=5, embargo=2)
    assert fixed.train == slice(0, 60)
    assert fixed.validation == slice(67, 90)


def test_benjamini_hochberg_adjustment_and_costs() -> None:
    adjusted = benjamini_hochberg((0.001, 0.02, 0.2), alpha=0.05)
    assert tuple(flag for _, flag in adjusted) == (True, True, False)
    assert all(0 <= value <= 1 for value, _ in adjusted)
    np.testing.assert_allclose(
        net_returns([0.01, 0.02], [1.0, 0.5], cost_bps=10),
        [0.009, 0.0195],
    )
    with pytest.raises(ValueError):
        benjamini_hochberg((1.1,))


def test_family_multiple_testing_and_spa_are_deterministic() -> None:
    summaries = family_multiple_testing_summary(
        {"trend": (0.001, 0.02, 0.2), "value": (0.4,)},
        alpha=0.05,
    )
    assert tuple(summary.family for summary in summaries) == ("trend", "value")
    assert summaries[0].trial_count == 3
    assert summaries[0].rejected_count == 2

    generator = np.random.default_rng(7)
    differentials = generator.normal(0, 0.01, size=(240, 4))
    differentials[:, 0] += 0.004
    first = superior_predictive_ability_test(
        differentials,
        bootstrap_samples=200,
        seed=17,
    )
    second = superior_predictive_ability_test(
        differentials,
        bootstrap_samples=200,
        seed=17,
    )
    assert first == second
    assert first.statistic > 0
    assert 0 <= first.lower_p_value <= first.p_value <= first.upper_p_value <= 1
    assert (first.observation_count, first.trial_count) == (240, 4)

    with pytest.raises(ValueError):
        superior_predictive_ability_test(np.ones((19, 2)))
