import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from aquant.domain.data_release import DataReleaseId
from aquant.factors.feature_sets import FactorMember, FeatureSetRegistry, FeatureSetSpec
from aquant.factors.lineage import validate_lineage
from aquant.factors.registry import FactorRegistry
from aquant.factors.spec import FactorLayer, FactorSpec, FactorStatus, SourceType
from aquant.factors.types import FactorResult, FactorValue

NOW = datetime(2026, 7, 28, tzinfo=UTC)
RELEASE = DataReleaseId("cn_equity_20260728_001")


def spec(**updates: object) -> FactorSpec:
    values: dict[str, object] = {
        "factor_id": "momentum_20d",
        "name": "Momentum 20D",
        "description": "Twenty-session close momentum.",
        "family": "momentum",
        "layer": FactorLayer.L2A,
        "version": "1.0.0",
        "status": FactorStatus.DRAFT,
        "hypothesis": "Medium-term price strength persists.",
        "expected_direction": 1,
        "expression": "RankCS(Delta(Close, 20))",
        "input_fields": ("Close",),
        "required_history": 20,
        "data_lag": 0,
        "universe": "all_a_share",
        "target_horizons": (20, 5),
        "preprocessing": ("winsorize_mad",),
        "neutralization": ("industry", "size"),
        "parameters": {"window": 20},
        "source_type": SourceType.FORMULA,
        "source_reference": "AQuant baseline",
        "tags": ("price", "medium_term"),
        "complexity_score": 3,
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(updates)
    return FactorSpec.model_validate(values)


def test_factor_spec_is_canonical_serializable_and_immutable() -> None:
    first = spec()
    equivalent = spec(
        factor_id="other",
        expression=" RankCS( (Delta(Close, 20)) ) ",
    )
    assert first.expression_hash == equivalent.expression_hash
    assert first.target_horizons == (5, 20)
    assert FactorSpec.from_json(first.to_json()) == first
    assert FactorSpec.from_yaml(first.to_yaml()) == first
    with pytest.raises(ValidationError):
        first.version = "2"  # type: ignore[misc]


@pytest.mark.parametrize(
    "updates",
    [
        {"implementation": "aquant.atomic.momentum", "expression": "Close"},
        {"expression": None, "implementation": None},
        {"required_history": -1},
        {"target_horizons": (0,)},
        {"valid_from": date(2026, 2, 1), "valid_to": date(2026, 1, 1)},
        {"parent_factor_ids": ("momentum_20d",)},
        {"expression_hash": "0" * 64},
    ],
)
def test_factor_spec_rejects_invalid_metadata(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        spec(**updates)


def test_registry_rejects_version_and_canonical_expression_conflicts(tmp_path: Path) -> None:
    database = tmp_path / "factors.sqlite3"
    first = spec()
    with FactorRegistry(database) as registry:
        registry.register(first)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(first)
        with pytest.raises(ValueError, match="canonical expression"):
            registry.register(
                spec(
                    factor_id="same_math",
                    expression="RankCS((Delta(Close,20)))",
                )
            )

    with FactorRegistry(database) as reloaded:
        assert reloaded.get(first.factor_id, first.version) == first


def test_registry_enforces_lifecycle_evidence_and_persists_status(tmp_path: Path) -> None:
    database = tmp_path / "factors.sqlite3"
    evidence = hashlib.sha256(b"walk-forward evidence").hexdigest()
    with FactorRegistry(database) as registry:
        registry.register(spec())
        registry.transition("momentum_20d", "1.0.0", FactorStatus.COMPUTED)
        with pytest.raises(ValueError, match="evidence"):
            registry.transition("momentum_20d", "1.0.0", FactorStatus.VALIDATED)
        registry.transition(
            "momentum_20d",
            "1.0.0",
            FactorStatus.VALIDATED,
            evidence_hash=evidence,
        )
        registry.transition("momentum_20d", "1.0.0", FactorStatus.APPROVED)
        assert registry.production_specs()[0].status == FactorStatus.APPROVED

    with FactorRegistry(database) as reloaded:
        assert reloaded.get("momentum_20d", "1.0.0").status == FactorStatus.APPROVED
        with pytest.raises(ValueError, match="invalid"):
            reloaded.transition("momentum_20d", "1.0.0", FactorStatus.DRAFT)


def test_lineage_rejects_unknown_parents_and_cycles() -> None:
    parent = spec()
    child = spec(
        factor_id="momentum_40d",
        expression="RankCS(Delta(Close, 40))",
        parent_factor_ids=("momentum_20d",),
        variant_dimension="window",
        parameters={"window": 40},
    )
    validate_lineage((parent, child))
    with pytest.raises(ValueError, match="unknown"):
        validate_lineage((child,))
    cyclic_parent = parent.model_copy(update={"parent_factor_ids": ("momentum_40d",)})
    with pytest.raises(ValueError, match="cycle"):
        validate_lineage((cyclic_parent, child))


def test_factor_result_requires_unique_stable_primary_key_order() -> None:
    first = FactorValue(
        date(2026, 7, 27),
        "000001.SZ",
        "momentum_20d",
        "1.0.0",
        0.1,
        True,
        (),
        RELEASE,
        NOW,
    )
    second = FactorValue(
        date(2026, 7, 28),
        "000001.SZ",
        "momentum_20d",
        "1.0.0",
        0.2,
        True,
        (),
        RELEASE,
        NOW,
    )
    assert FactorResult((first, second)).values == (first, second)
    with pytest.raises(ValueError, match="unique"):
        FactorResult((first, first))
    with pytest.raises(ValueError, match="ordering"):
        FactorResult((second, first))


def test_feature_set_pins_factor_versions() -> None:
    feature_set = FeatureSetSpec(
        feature_set_id="baseline_simple",
        version="1.0.0",
        description="Audited baseline",
        target_horizon=20,
        universe="all_a_share",
        factor_members=(FactorMember(factor_id="momentum_20d", factor_version="1.0.0"),),
        selection_method="approved_baseline",
        created_from_experiment="baseline_20260728",
        data_release_id=RELEASE,
    )
    assert feature_set.factor_members[0].factor_version == "1.0.0"
    with pytest.raises(ValidationError, match="duplicate"):
        FeatureSetSpec.model_validate(
            {**feature_set.model_dump(), "factor_members": feature_set.factor_members * 2}
        )


def test_feature_set_content_hash_and_atomic_registry_are_idempotent(tmp_path: Path) -> None:
    feature_set = FeatureSetSpec(
        feature_set_id="baseline_raw",
        version="1.0.0",
        description="Audited raw baseline",
        target_horizon=10,
        universe="all_a_share",
        factor_members=(FactorMember(factor_id="momentum_20d", factor_version="1.0.0"),),
        selection_method="production_gate",
        created_from_experiment="long_horizon_20260728",
        data_release_id=RELEASE,
        effective_from=date(2026, 8, 1),
        training_window_days=1260,
        config_hash=hashlib.sha256(b"feature-config").hexdigest(),
    )
    registry = FeatureSetRegistry(tmp_path)
    first = registry.publish(feature_set)
    second = registry.publish(feature_set)
    assert first == second
    assert FeatureSetSpec.model_validate_json(first.read_text()).content_hash == (
        feature_set.content_hash
    )
    changed = feature_set.model_copy(
        update={
            "factor_members": (FactorMember(factor_id="value", factor_version="1.0.0"),),
            "content_hash": "",
        }
    )
    changed = FeatureSetSpec.model_validate(changed.model_dump())
    with pytest.raises(ValueError, match="different content"):
        registry.publish(changed)
