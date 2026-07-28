import math
from pathlib import Path

import pytest

from aquant.factors.atomic import baseline_factor_library
from aquant.factors.evaluation import (
    ProductionEvidence,
    ProductionGateConfig,
    evaluate_production_gate,
)
from aquant.factors.exceptions import FactorRegistryError
from aquant.factors.registry import FactorRegistry
from aquant.factors.spec import FactorStatus


def _config() -> ProductionGateConfig:
    return ProductionGateConfig.from_yaml(Path("config/factors/production_gate_v1.yaml"))


def _passing_evidence(**updates: object) -> ProductionEvidence:
    values: dict[str, object] = {
        "history_years": 10.0,
        "cross_sections": 2000,
        "missing_rate": 0.05,
        "oos_rank_ic": 0.03,
        "rank_icir": 0.4,
        "directional_ic_win_rate": 0.58,
        "positive_year_ratio": 0.8,
        "monotonicity": 0.8,
        "turnover": 0.3,
        "directional_net_return": 0.001,
        "maximum_peer_correlation": 0.7,
        "conditional_rank_ic": 0.01,
        "complexity": 4.0,
        "maximum_style_exposure": 0.5,
        "extreme_regime_rank_ic": 0.01,
    }
    values.update(updates)
    return ProductionEvidence(**values)  # type: ignore[arg-type]


def test_configured_gate_scores_components_and_hard_vetoes() -> None:
    config = _config()
    passing = evaluate_production_gate(_passing_evidence(), config)
    assert passing.passed
    assert len(passing.config_hash) == len(passing.evidence_hash) == 64
    assert dict(passing.components)["predictive"] == pytest.approx(0.03)

    leakage = evaluate_production_gate(
        _passing_evidence(leakage_detected=True),
        config,
    )
    assert not leakage.passed
    assert "FUTURE_LEAKAGE" in leakage.rejections

    redundant = evaluate_production_gate(
        _passing_evidence(
            maximum_peer_correlation=0.95,
            conditional_rank_ic=0.001,
        ),
        config,
    )
    assert not redundant.passed
    assert "REDUNDANCY_FAILURE" in redundant.rejections

    non_finite = evaluate_production_gate(
        _passing_evidence(oos_rank_ic=float("nan")),
        config,
    )
    assert not non_finite.passed
    assert "NON_FINITE_EVIDENCE" in non_finite.rejections
    assert math.isfinite(non_finite.overall_score)


def test_gate_rejects_invalid_threshold_configuration() -> None:
    payload = _config().model_dump()
    payload["maximum_missing_rate"] = 2
    with pytest.raises(ValueError, match="between zero and one"):
        ProductionGateConfig.model_validate(payload)


def test_registry_requires_passing_gate_for_production(tmp_path: Path) -> None:
    factor = baseline_factor_library()[0]
    decision = evaluate_production_gate(_passing_evidence(), _config())
    with FactorRegistry(tmp_path / "registry.sqlite3") as registry:
        registry.register(factor.spec)
        registry.transition(factor.spec.factor_id, factor.spec.version, FactorStatus.COMPUTED)
        registry.transition(
            factor.spec.factor_id,
            factor.spec.version,
            FactorStatus.VALIDATED,
            evidence_hash=decision.evidence_hash,
        )
        registry.transition(factor.spec.factor_id, factor.spec.version, FactorStatus.APPROVED)
        with pytest.raises(FactorRegistryError, match="gate"):
            registry.transition(
                factor.spec.factor_id,
                factor.spec.version,
                FactorStatus.PRODUCTION,
            )
        production = registry.transition(
            factor.spec.factor_id,
            factor.spec.version,
            FactorStatus.PRODUCTION,
            gate_decision=decision,
        )
        assert production.status is FactorStatus.PRODUCTION
