import hashlib

import numpy as np
import pytest

from aquant.factors.evaluation import FactorApprovalStatus
from aquant.factors.mining import (
    CandidateGenerator,
    CandidateSelector,
    WalkForwardSplit,
    benjamini_hochberg,
)
from aquant.factors.registry import FactorRegistry, RegisteredFactor


def test_candidate_generation_is_deterministic_valid_and_deduplicated() -> None:
    generator = CandidateGenerator()
    enumerated = generator.enumerate_templates(fields=("Close",), windows=(5, 10))
    first = generator.random_search(fields=("Close", "Volume"), windows=(5, 10), count=5, seed=7)
    second = generator.random_search(fields=("Close", "Volume"), windows=(5, 10), count=5, seed=7)
    assert len(enumerated) == 6
    assert [item.candidate_id for item in first] == [item.candidate_id for item in second]
    assert len({item.candidate_id for item in first}) == len(first)
    assert all(item.expression.complexity <= 64 for item in (*enumerated, *first))
    with pytest.raises(ValueError):
        generator.random_search(fields=(), windows=(5,), count=1, seed=1)


def test_walk_forward_evaluation_uses_strictly_later_validation() -> None:
    candidate = CandidateGenerator().enumerate_templates(fields=("Close",), windows=(5,))[0]
    factors = np.tile(np.arange(10, dtype=float), (8, 1))
    returns = factors * 0.01
    split = WalkForwardSplit(0, 4, 4, 8)
    evaluation = CandidateSelector().evaluate(candidate, factors, returns, split)
    assert evaluation.train.rank_ic_mean == pytest.approx(1)
    assert evaluation.validation.rank_ic_mean == pytest.approx(1)
    assert np.isfinite(evaluation.score)
    with pytest.raises(ValueError, match="ordered"):
        WalkForwardSplit(0, 5, 4, 8)


def test_correlation_filter_keeps_best_non_redundant_candidate() -> None:
    candidates = CandidateGenerator().enumerate_templates(fields=("Close",), windows=(5, 10))
    factors = np.tile(np.arange(10, dtype=float), (8, 1))
    returns = factors * 0.01
    selector = CandidateSelector()
    split = WalkForwardSplit(0, 4, 4, 8)
    first = selector.evaluate(candidates[0], factors, returns, split)
    second = selector.evaluate(candidates[1], factors, returns, split)
    accepted = selector.correlation_filter(
        ((first, factors), (second, factors * 2)), maximum_absolute_correlation=0.8
    )
    assert len(accepted) == 1
    with pytest.raises(ValueError, match="threshold"):
        selector.correlation_filter((), maximum_absolute_correlation=2)


def test_benjamini_hochberg_controls_multiple_testing() -> None:
    assert benjamini_hochberg((0.001, 0.02, 0.2), alpha=0.05) == (True, True, False)
    assert benjamini_hochberg(()) == ()
    with pytest.raises(ValueError):
        benjamini_hochberg((1.1,))


def test_factor_lifecycle_requires_evidence_and_only_exposes_approved() -> None:
    candidate = CandidateGenerator().enumerate_templates(fields=("Close",), windows=(5,))[0]
    registry = FactorRegistry()
    record = RegisteredFactor("momentum", "1.0.0", candidate)
    registry.register(record)
    registry.transition("momentum", "1.0.0", FactorApprovalStatus.COMPUTED)
    evidence = hashlib.sha256(b"walk-forward-report").hexdigest()
    with pytest.raises(ValueError, match="evidence"):
        registry.transition("momentum", "1.0.0", FactorApprovalStatus.VALIDATED)
    registry.transition("momentum", "1.0.0", FactorApprovalStatus.VALIDATED, evidence_hash=evidence)
    registry.transition("momentum", "1.0.0", FactorApprovalStatus.APPROVED, evidence_hash=evidence)
    assert registry.production_eligible()[0].factor_id == "momentum"
    registry.transition("momentum", "1.0.0", FactorApprovalStatus.PRODUCTION)
    registry.transition("momentum", "1.0.0", FactorApprovalStatus.DEPRECATED)
    assert registry.production_eligible() == ()


def test_factor_registry_rejects_duplicates_and_skipped_transition() -> None:
    candidate = CandidateGenerator().enumerate_templates(fields=("Close",), windows=(5,))[0]
    registry = FactorRegistry()
    record = RegisteredFactor("x", "1", candidate)
    registry.register(record)
    with pytest.raises(ValueError, match="already"):
        registry.register(record)
    with pytest.raises(ValueError, match="invalid"):
        registry.transition("x", "1", FactorApprovalStatus.APPROVED)
