from datetime import date

import pytest

from aquant.domain.data_release import DataReleaseId
from aquant.factors.atomic import baseline_factor_library
from aquant.factors.feature_sets import (
    FeatureRole,
    FeatureSetStatus,
    build_baseline_feature_sets,
)


def test_baseline_feature_sets_pin_members_and_processing() -> None:
    factors = tuple(factor.spec for factor in baseline_factor_library())
    compact = tuple(factor.factor_id for factor in factors[:20])
    result = build_baseline_feature_sets(
        factors,
        compact,
        data_release_id=DataReleaseId("cn_equity_20260717_001"),
        effective_from=date(2026, 7, 17),
        created_from_experiment="five_year_convergence_v1",
        code_version="abc123",
    )
    assert len(result.raw.factor_members) == 73
    assert len(result.compact.factor_members) == 20
    assert result.neutral.factor_members == result.compact.factor_members
    assert result.raw.neutralization == ()
    assert result.neutral.neutralization == ("pit_industry", "log_float_market_cap")
    assert result.raw.status == FeatureSetStatus.DRAFT
    assert result.compact.status == FeatureSetStatus.VALIDATED
    assert result.neutral.status == FeatureSetStatus.DRAFT
    assert (
        next(
            member
            for member in result.raw.factor_members
            if member.factor_id == "log_total_market_cap"
        ).role
        == FeatureRole.RISK_CONTROL
    )
    assert len({item.content_hash for item in (result.raw, result.compact, result.neutral)}) == 3


@pytest.mark.parametrize("count", [14, 26])
def test_baseline_feature_sets_reject_uncontrolled_compact_size(count: int) -> None:
    factors = tuple(factor.spec for factor in baseline_factor_library())
    with pytest.raises(ValueError, match="15 to 25"):
        build_baseline_feature_sets(
            factors,
            tuple(factor.factor_id for factor in factors[:count]),
            data_release_id=DataReleaseId("cn_equity_20260717_001"),
            effective_from=date(2026, 7, 17),
            created_from_experiment="invalid",
        )
