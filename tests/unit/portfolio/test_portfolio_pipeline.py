from datetime import UTC, date, datetime
from decimal import Decimal

import numpy as np
import pytest

from aquant.domain.identifiers import Symbol
from aquant.portfolio import (
    ConstrainedPortfolioOptimizer,
    MultiFactorScorer,
    PortfolioConstraints,
    calculate_risk_report,
    construct_top_n_equal_weight,
)

SYMBOLS = tuple(Symbol.parse(value) for value in ("600000.XSHG", "600001.XSHG", "000001.XSHE"))


def test_multifactor_scoring_standardizes_and_combines_complete_coverage() -> None:
    scorer = MultiFactorScorer()
    scores = scorer.score(
        {
            "momentum": dict(zip(SYMBOLS, (1.0, 2.0, 3.0), strict=True)),
            "value": dict(zip(SYMBOLS, (3.0, 2.0, 1.0), strict=True)),
        },
        {"momentum": 1.0, "value": -0.5},
    )
    assert scores[0].symbol == SYMBOLS[2]
    assert scores[0].score > scores[-1].score
    with pytest.raises(ValueError):
        scorer.score({}, {})
    with pytest.raises(ValueError):
        scorer.score({"x": {SYMBOLS[0]: 1}}, {"y": 1})


def test_top_n_constructor_reserves_cash_and_caps_name_weight() -> None:
    scores = MultiFactorScorer().score(
        {"x": dict(zip(SYMBOLS, (1.0, 2.0, 3.0), strict=True))}, {"x": 1.0}
    )
    portfolio = construct_top_n_equal_weight(
        scores,
        top_n=2,
        maximum_weight=Decimal("0.4"),
        minimum_cash_weight=Decimal("0.1"),
        strategy_id="multi-factor",
        trade_date=date(2026, 7, 17),
        asof_time=datetime(2026, 7, 16, 7, tzinfo=UTC),
        data_release_id="cn_equity_20260716_001",
        signal_version="v1",
    )
    assert len(portfolio.positions) == 2
    assert all(position.target_weight == Decimal("0.4") for position in portfolio.positions)
    assert portfolio.cash_weight == Decimal("0.2")


def test_constrained_optimizer_obeys_name_cash_turnover_and_industry_rules() -> None:
    optimizer = ConstrainedPortfolioOptimizer()
    constraints = PortfolioConstraints(
        maximum_weight=0.4,
        maximum_turnover=0.5,
        minimum_cash_weight=0.1,
        maximum_industry_deviation=0.2,
    )
    current = np.array([0.2, 0.2, 0.2])
    benchmark = np.array([0.3, 0.3, 0.3])
    result = optimizer.optimize(
        symbols=SYMBOLS,
        scores=np.array([3.0, 2.0, 1.0]),
        current_weights=current,
        benchmark_weights=benchmark,
        industries=("bank", "bank", "tech"),
        constraints=constraints,
    )
    weights = np.array([result.weights[symbol] for symbol in SYMBOLS])
    assert np.all(weights >= -1e-8)
    assert np.max(weights) <= 0.4 + 1e-6
    assert result.cash_weight >= 0.1 - 1e-6
    assert result.turnover <= 0.5 + 1e-6
    bank_active = weights[:2].sum() - benchmark[:2].sum()
    assert abs(bank_active) <= 0.2 + 1e-6


def test_risk_report_reconciles_weights_and_active_exposure() -> None:
    weights = {SYMBOLS[0]: 0.4, SYMBOLS[2]: 0.3}
    current = {SYMBOLS[0]: 0.2, SYMBOLS[1]: 0.2, SYMBOLS[2]: 0.2}
    benchmark = {symbol: 0.3 for symbol in SYMBOLS}
    industries = {SYMBOLS[0]: "bank", SYMBOLS[1]: "bank", SYMBOLS[2]: "tech"}
    report = calculate_risk_report(
        weights,
        current_weights=current,
        benchmark_weights=benchmark,
        industries=industries,
    )
    assert report.maximum_name_weight == 0.4
    assert report.cash_weight == pytest.approx(0.3)
    assert report.industry_weights["bank"] == 0.4
    assert report.active_industry_weights["bank"] == pytest.approx(-0.2)
    assert report.turnover == pytest.approx(0.25)
    with pytest.raises(ValueError, match="incomplete"):
        calculate_risk_report(weights, current_weights={}, benchmark_weights={}, industries={})


def test_constraints_and_optimizer_validate_inputs() -> None:
    with pytest.raises(ValueError):
        PortfolioConstraints(maximum_weight=0)
    with pytest.raises(ValueError, match="shapes"):
        ConstrainedPortfolioOptimizer().optimize(
            symbols=SYMBOLS,
            scores=np.ones(2),
            current_weights=np.ones(3),
            benchmark_weights=np.ones(3),
            industries=("a", "b", "c"),
            constraints=PortfolioConstraints(),
        )
