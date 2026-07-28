import hashlib
import json
from dataclasses import dataclass
from datetime import date

from aquant.domain.data_release import DataReleaseId
from aquant.factors.feature_sets.spec import FactorMember, FeatureSetSpec
from aquant.factors.spec import FactorSpec


@dataclass(frozen=True, slots=True)
class BaselineFeatureSets:
    raw: FeatureSetSpec
    compact: FeatureSetSpec
    neutral: FeatureSetSpec


def build_baseline_feature_sets(
    factors: tuple[FactorSpec, ...],
    compact_factor_ids: tuple[str, ...],
    *,
    data_release_id: DataReleaseId,
    effective_from: date,
    created_from_experiment: str,
    target_horizon: int = 20,
    training_window_days: int = 1260,
    code_version: str = "working-tree",
) -> BaselineFeatureSets:
    """Build the three pinned baselines after evidence-based convergence."""
    if len(compact_factor_ids) < 15 or len(compact_factor_ids) > 25:
        raise ValueError("compact baseline must contain 15 to 25 factors")
    by_id = {factor.factor_id: factor for factor in factors}
    if len(by_id) != len(factors):
        raise ValueError("baseline factors must have unique factor ids")
    missing = set(compact_factor_ids) - set(by_id)
    if missing:
        raise ValueError(f"compact baseline contains unknown factors: {sorted(missing)}")
    if len(set(compact_factor_ids)) != len(compact_factor_ids):
        raise ValueError("compact baseline must not contain duplicate factor ids")

    all_members = _members(tuple(by_id[factor_id] for factor_id in sorted(by_id)))
    compact_members = _members(tuple(by_id[factor_id] for factor_id in sorted(compact_factor_ids)))
    common = {
        "version": "1.0.0",
        "target_horizon": target_horizon,
        "universe": "all_a_share",
        "created_from_experiment": created_from_experiment,
        "data_release_id": data_release_id,
        "effective_from": effective_from,
        "training_window_days": training_window_days,
        "code_version": code_version,
        "status": "DRAFT",
        "winsorization_method": "mad_5",
        "missing_value_strategy": "preserve",
    }
    raw = _spec(
        feature_set_id="baseline_raw_v1",
        description="All version-pinned baseline factors without neutralization",
        members=all_members,
        selection_method="baseline_registry_snapshot",
        standardization="cross_sectional_zscore",
        neutralization=(),
        common=common,
    )
    compact = _spec(
        feature_set_id="baseline_compact_v1",
        description="Evidence-selected cluster representatives and residual candidates",
        members=compact_members,
        selection_method="three_correlation_convergence",
        standardization="cross_sectional_zscore",
        neutralization=(),
        common=common,
    )
    neutral = _spec(
        feature_set_id="baseline_neutral_v1",
        description="Compact factors neutralized by PIT industry and log float market cap",
        members=compact_members,
        selection_method="three_correlation_convergence",
        standardization="cross_sectional_zscore",
        neutralization=("pit_industry", "log_float_market_cap"),
        common=common,
    )
    return BaselineFeatureSets(raw, compact, neutral)


def _members(factors: tuple[FactorSpec, ...]) -> tuple[FactorMember, ...]:
    return tuple(
        FactorMember(factor_id=factor.factor_id, factor_version=factor.version)
        for factor in factors
    )


def _spec(
    *,
    feature_set_id: str,
    description: str,
    members: tuple[FactorMember, ...],
    selection_method: str,
    standardization: str,
    neutralization: tuple[str, ...],
    common: dict[str, object],
) -> FeatureSetSpec:
    config = {
        "feature_set_id": feature_set_id,
        "members": [member.model_dump() for member in members],
        "standardization": standardization,
        "winsorization": common["winsorization_method"],
        "missing": common["missing_value_strategy"],
        "neutralization": neutralization,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return FeatureSetSpec(
        feature_set_id=feature_set_id,
        description=description,
        factor_members=members,
        preprocessing=("winsorize", "standardize"),
        neutralization=neutralization,
        selection_method=selection_method,
        standardization_method=standardization,
        config_hash=config_hash,
        **common,
    )
