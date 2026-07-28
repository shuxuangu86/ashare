from datetime import UTC, datetime

import numpy as np
import pytest

from aquant.factors.atomic import baseline_factor_library
from aquant.factors.generation import generate_window_variants
from aquant.factors.generation.deduplication import numerical_duplicates
from aquant.factors.selection.clustering import (
    correlation_matrix,
    hierarchical_clusters,
    representative_by_score,
    residual_information,
)


def test_generation_is_sparse_budgeted_single_dimension_and_audited() -> None:
    parent = baseline_factor_library()[10].spec
    batch = generate_window_variants(
        parent,
        template="Rank(Delta(Close, {window}))",
        windows=(5, 20, 60),
        batch_id="momentum_windows_001",
        family_budget="momentum_reversal",
        created_at=datetime(2026, 7, 28, tzinfo=UTC),
    )
    assert batch.attempted == 3
    assert len(batch.candidates) == 3
    assert not batch.failures
    assert all(item.variant_dimension == "window" for item in batch.candidates)
    assert all(item.parent_factor_ids == (parent.factor_id,) for item in batch.candidates)
    assert len({item.expression_hash for item in batch.candidates}) == 3
    with pytest.raises(ValueError, match="budget"):
        generate_window_variants(
            parent,
            template="Rank(Delta(Close, {window}))",
            windows=tuple(range(1, 122)),
            batch_id="too_many",
            family_budget="momentum_reversal",
            created_at=datetime(2026, 7, 28, tzinfo=UTC),
        )


def test_exact_numeric_and_high_correlation_duplicates_are_identified() -> None:
    values = np.column_stack(
        (np.arange(20.0), np.arange(20.0), np.arange(20.0) * 2, np.sin(np.arange(20.0)))
    )
    duplicates = numerical_duplicates(values, correlation_threshold=0.99)
    assert (0, 1) in duplicates
    assert (0, 2) in duplicates
    assert (0, 3) not in duplicates


def test_clustering_selects_representative_but_preserves_residual_information() -> None:
    generator = np.random.default_rng(7)
    base = generator.normal(size=200)
    values = np.column_stack((base, base + generator.normal(scale=0.01, size=200), -base))
    correlations = correlation_matrix(values)
    clusters = hierarchical_clusters(correlations, maximum_distance=0.1)
    assert len(set(clusters)) == 1
    representative = representative_by_score(clusters, [1.0, 2.0, 0.5])
    assert representative[clusters[0]] == 1
    residual = residual_information(values[:, 0], values[:, 1])
    assert np.isfinite(residual).all()
    assert abs(np.corrcoef(residual, values[:, 1])[0, 1]) < 1e-10
