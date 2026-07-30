from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from aquant.factors.spec import FactorStatus


class EligibilityReason(StrEnum):
    PASSED_RESEARCH_SAFETY = "PASSED_RESEARCH_SAFETY"
    PASSED_FEATURE_ELIGIBILITY = "PASSED_FEATURE_ELIGIBILITY"
    FUTURE_DATA_LEAKAGE = "FUTURE_DATA_LEAKAGE"
    NON_REPRODUCIBLE = "NON_REPRODUCIBLE"
    FORMULA_IMPLEMENTATION_ERROR = "FORMULA_IMPLEMENTATION_ERROR"
    DATA_DEPENDENCY_MISSING = "DATA_DEPENDENCY_MISSING"
    DEGENERATE_OUTPUT = "DEGENERATE_OUTPUT"
    COVERAGE_TOO_LOW = "COVERAGE_TOO_LOW"
    OOS_DIRECTIONAL_COLLAPSE = "OOS_DIRECTIONAL_COLLAPSE"
    EXACT_DUPLICATE = "EXACT_DUPLICATE"
    UNACCEPTABLE_INSTABILITY = "UNACCEPTABLE_INSTABILITY"
    FAMILY_PARETO = "FAMILY_PARETO"
    FAMILY_STRONGEST = "FAMILY_STRONGEST"
    FAMILY_MOST_STABLE = "FAMILY_MOST_STABLE"
    FAMILY_LOWEST_TURNOVER = "FAMILY_LOWEST_TURNOVER"
    FAMILY_MOST_INDEPENDENT = "FAMILY_MOST_INDEPENDENT"
    FAMILY_REGIME_COMPLEMENT = "FAMILY_REGIME_COMPLEMENT"


@dataclass(frozen=True, slots=True)
class ResearchSafetyEvidence:
    formula_verified: bool
    dependencies_available: bool
    leakage_free: bool
    reproducible: bool
    non_degenerate: bool
    coverage: float


@dataclass(frozen=True, slots=True)
class FeatureEvidence:
    oos_rank_ic: float
    oos_rank_icir: float
    annual_stability: float
    monotonicity: float
    coverage: float
    turnover: float
    cost_sensitivity: float
    complexity: float
    max_peer_correlation: float
    regime_complementarity: float
    exact_duplicate: bool = False


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    tier: FactorStatus
    passed: bool
    reason_codes: tuple[EligibilityReason, ...]


def evaluate_research_safety(
    evidence: ResearchSafetyEvidence,
    *,
    minimum_coverage: float = 0.6,
) -> EligibilityDecision:
    failures: list[EligibilityReason] = []
    if not evidence.formula_verified:
        failures.append(EligibilityReason.FORMULA_IMPLEMENTATION_ERROR)
    if not evidence.dependencies_available:
        failures.append(EligibilityReason.DATA_DEPENDENCY_MISSING)
    if not evidence.leakage_free:
        failures.append(EligibilityReason.FUTURE_DATA_LEAKAGE)
    if not evidence.reproducible:
        failures.append(EligibilityReason.NON_REPRODUCIBLE)
    if not evidence.non_degenerate:
        failures.append(EligibilityReason.DEGENERATE_OUTPUT)
    if not np.isfinite(evidence.coverage) or evidence.coverage < minimum_coverage:
        failures.append(EligibilityReason.COVERAGE_TOO_LOW)
    return EligibilityDecision(
        FactorStatus.REJECTED if failures else FactorStatus.RESEARCH_VALIDATED,
        not failures,
        tuple(failures) or (EligibilityReason.PASSED_RESEARCH_SAFETY,),
    )


def evaluate_feature_eligibility(
    evidence: FeatureEvidence,
    *,
    maximum_oos_collapse: float = -0.03,
    maximum_peer_correlation: float = 0.999,
    minimum_coverage: float = 0.6,
) -> EligibilityDecision:
    failures: list[EligibilityReason] = []
    if evidence.exact_duplicate or evidence.max_peer_correlation >= maximum_peer_correlation:
        failures.append(EligibilityReason.EXACT_DUPLICATE)
    if evidence.oos_rank_ic < maximum_oos_collapse:
        failures.append(EligibilityReason.OOS_DIRECTIONAL_COLLAPSE)
    if evidence.coverage < minimum_coverage or evidence.annual_stability < -0.5:
        failures.append(EligibilityReason.UNACCEPTABLE_INSTABILITY)
    return EligibilityDecision(
        FactorStatus.RESEARCH_VALIDATED if failures else FactorStatus.FEATURE_ELIGIBLE,
        not failures,
        tuple(failures) or (EligibilityReason.PASSED_FEATURE_ELIGIBILITY,),
    )


def family_diversity_selection(
    evidence: dict[str, FeatureEvidence],
    *,
    maximum_members: int = 12,
) -> dict[str, tuple[EligibilityReason, ...]]:
    """Select Pareto/archetype representatives without imposing the production alpha gate."""
    if maximum_members < 5:
        raise ValueError("family selection must allow the five required archetypes")
    if not evidence:
        return {}
    eligible = {
        factor_id: metrics
        for factor_id, metrics in evidence.items()
        if evaluate_feature_eligibility(metrics).passed
    }
    if not eligible:
        return {}
    selected: dict[str, list[EligibilityReason]] = {}

    def keep(factor_id: str, reason: EligibilityReason) -> None:
        selected.setdefault(factor_id, []).append(reason)

    keep(
        max(eligible, key=lambda item: abs(eligible[item].oos_rank_ic)),
        EligibilityReason.FAMILY_STRONGEST,
    )
    keep(
        max(eligible, key=lambda item: eligible[item].annual_stability),
        EligibilityReason.FAMILY_MOST_STABLE,
    )
    keep(
        min(eligible, key=lambda item: eligible[item].turnover),
        EligibilityReason.FAMILY_LOWEST_TURNOVER,
    )
    keep(
        min(eligible, key=lambda item: eligible[item].max_peer_correlation),
        EligibilityReason.FAMILY_MOST_INDEPENDENT,
    )
    keep(
        max(eligible, key=lambda item: eligible[item].regime_complementarity),
        EligibilityReason.FAMILY_REGIME_COMPLEMENT,
    )
    for factor_id in sorted(
        _pareto_front(eligible),
        key=lambda item: _utility(eligible[item]),
        reverse=True,
    ):
        if len(selected) >= maximum_members:
            break
        keep(factor_id, EligibilityReason.FAMILY_PARETO)
    return {factor_id: tuple(reasons) for factor_id, reasons in sorted(selected.items())}


def _pareto_front(evidence: dict[str, FeatureEvidence]) -> tuple[str, ...]:
    vectors = {
        factor_id: np.asarray(
            [
                abs(item.oos_rank_ic),
                item.oos_rank_icir,
                item.annual_stability,
                item.monotonicity,
                item.coverage,
                -item.turnover,
                -item.cost_sensitivity,
                -item.complexity,
                -item.max_peer_correlation,
                item.regime_complementarity,
            ]
        )
        for factor_id, item in evidence.items()
    }
    return tuple(
        factor_id
        for factor_id, vector in vectors.items()
        if not any(
            other_id != factor_id and np.all(other >= vector) and np.any(other > vector)
            for other_id, other in vectors.items()
        )
    )


def _utility(item: FeatureEvidence) -> float:
    return (
        abs(item.oos_rank_ic)
        + 0.5 * item.oos_rank_icir
        + 0.2 * item.annual_stability
        + 0.1 * item.coverage
        - 0.05 * item.turnover
        - 0.01 * item.complexity
        - 0.1 * item.max_peer_correlation
        + 0.2 * item.regime_complementarity
    )
