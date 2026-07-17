from dataclasses import dataclass

from aquant.domain.identifiers import Symbol


@dataclass(frozen=True, slots=True)
class PortfolioRiskReport:
    maximum_name_weight: float
    industry_weights: dict[str, float]
    active_industry_weights: dict[str, float]
    turnover: float
    cash_weight: float


def calculate_risk_report(
    weights: dict[Symbol, float],
    *,
    current_weights: dict[Symbol, float],
    benchmark_weights: dict[Symbol, float],
    industries: dict[Symbol, str],
) -> PortfolioRiskReport:
    universe = set(weights) | set(current_weights) | set(benchmark_weights)
    if any(symbol not in industries for symbol in universe):
        raise ValueError("industry mapping is incomplete")
    industry_weights: dict[str, float] = {}
    benchmark_industries: dict[str, float] = {}
    for symbol in universe:
        industry = industries[symbol]
        industry_weights[industry] = industry_weights.get(industry, 0.0) + weights.get(symbol, 0.0)
        benchmark_industries[industry] = benchmark_industries.get(
            industry, 0.0
        ) + benchmark_weights.get(symbol, 0.0)
    active = {
        industry: industry_weights.get(industry, 0.0) - benchmark_industries.get(industry, 0.0)
        for industry in set(industry_weights) | set(benchmark_industries)
    }
    turnover = (
        sum(abs(weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0)) for symbol in universe)
        / 2
    )
    return PortfolioRiskReport(
        max(weights.values(), default=0.0),
        industry_weights,
        active,
        turnover,
        1 - sum(weights.values()),
    )
