import numpy as np
import pytest

from aquant.domain.data_release import DataReleaseId
from aquant.factors import (
    FactorAnalyzer,
    FactorApprovalStatus,
    FactorCard,
    classic_factor_library,
)


def _monotonic_data() -> tuple[np.ndarray, np.ndarray]:
    factors = np.tile(np.arange(10, dtype=float), (4, 1))
    returns = factors * 0.01 - 0.02
    return factors, returns


def test_factor_analysis_reports_ic_groups_monotonicity_and_turnover() -> None:
    factors, returns = _monotonic_data()
    evaluation = FactorAnalyzer().evaluate(factors, returns, quantiles=5)
    assert np.allclose(evaluation.ic_by_period, 1.0)
    assert np.allclose(evaluation.rank_ic_by_period, 1.0)
    assert evaluation.ic_mean == pytest.approx(1.0)
    assert evaluation.rank_ic_mean == pytest.approx(1.0)
    assert evaluation.long_short_return > 0
    assert evaluation.monotonicity == pytest.approx(1.0)
    assert evaluation.turnover == 0
    assert evaluation.coverage == 1.0
    assert list(evaluation.quantile_returns) == sorted(evaluation.quantile_returns)


def test_analysis_handles_missing_values_and_measures_coverage() -> None:
    factors, returns = _monotonic_data()
    factors[0, :5] = np.nan
    returns[1, 0] = np.nan
    evaluation = FactorAnalyzer().evaluate(factors, returns)
    assert evaluation.coverage == 0.875
    assert len(evaluation.rank_ic_by_period) == 4


def test_decay_is_sorted_and_rejects_invalid_inputs() -> None:
    factors, returns = _monotonic_data()
    decay = FactorAnalyzer().decay_rank_ic(factors, {5: returns * 0.5, 1: returns})
    assert decay == ((1, pytest.approx(1.0)), (5, pytest.approx(1.0)))
    with pytest.raises(ValueError, match="at least"):
        FactorAnalyzer().decay_rank_ic(factors, {})
    with pytest.raises(ValueError, match="positive"):
        FactorAnalyzer().decay_rank_ic(factors, {0: returns})


@pytest.mark.parametrize(
    ("factor", "returns", "quantiles"),
    [
        (np.array([1.0]), np.array([1.0]), 5),
        (np.ones((2, 2)), np.ones((2, 3)), 5),
        (np.ones((2, 2)), np.ones((2, 2)), 1),
    ],
)
def test_analysis_validates_shapes_and_quantiles(
    factor: np.ndarray, returns: np.ndarray, quantiles: int
) -> None:
    with pytest.raises(ValueError):
        FactorAnalyzer().evaluate(factor, returns, quantiles=quantiles)


def test_factor_card_is_traceable_and_renders_markdown() -> None:
    factors, returns = _monotonic_data()
    evaluation = FactorAnalyzer().evaluate(factors, returns)
    spec = classic_factor_library()[0]
    card = FactorCard(
        name=spec.name,
        version=spec.version,
        hypothesis=spec.hypothesis,
        expression="RankCS(Sub(Div(Close, Ref(Close, 20)), 1))",
        expression_hash=spec.expression.expression_hash,
        data_sources=("akshare", "baostock"),
        available_time_rule="available_at <= asof_time",
        preprocessing="MAD winsorize, z-score, industry neutralize",
        expected_exposures="momentum and market beta",
        failure_conditions="coverage below 80% or cross-source gate failure",
        code_version="deadbeef",
        data_release_id=DataReleaseId("cn_equity_20260716_001"),
        approval_status=FactorApprovalStatus.VALIDATED,
        evaluation=evaluation,
    )
    markdown = card.to_markdown()
    assert "Factor Card: momentum_20d" in markdown
    assert "cn_equity_20260716_001" in markdown
    assert "akshare, baostock" in markdown
    assert "Rank IC mean" in markdown


def test_factor_card_rejects_incomplete_lineage() -> None:
    factors, returns = _monotonic_data()
    evaluation = FactorAnalyzer().evaluate(factors, returns)
    with pytest.raises(ValueError):
        FactorCard(
            name="",
            version="1",
            hypothesis="h",
            expression="Close",
            expression_hash="bad",
            data_sources=(),
            available_time_rule="pit",
            preprocessing="none",
            expected_exposures="none",
            failure_conditions="none",
            code_version="abc",
            data_release_id=DataReleaseId("cn_equity_20260716_001"),
            approval_status=FactorApprovalStatus.DRAFT,
            evaluation=evaluation,
        )
