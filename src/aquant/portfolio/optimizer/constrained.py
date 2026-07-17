import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from aquant.domain.identifiers import Symbol

Array = npt.NDArray[np.float64]
cp: Any = importlib.import_module("cvxpy")


@dataclass(frozen=True, slots=True)
class PortfolioConstraints:
    maximum_weight: float = 0.05
    maximum_turnover: float = 0.20
    minimum_cash_weight: float = 0.02
    maximum_industry_deviation: float = 0.05
    risk_aversion: float = 0.01

    def __post_init__(self) -> None:
        if not (
            0 < self.maximum_weight <= 1
            and 0 <= self.maximum_turnover <= 1
            and 0 <= self.minimum_cash_weight < 1
            and 0 <= self.maximum_industry_deviation <= 1
            and self.risk_aversion >= 0
        ):
            raise ValueError("portfolio constraints are invalid")


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    weights: dict[Symbol, float]
    cash_weight: float
    turnover: float
    objective_value: float


class ConstrainedPortfolioOptimizer:
    def optimize(
        self,
        *,
        symbols: tuple[Symbol, ...],
        scores: Array,
        current_weights: Array,
        benchmark_weights: Array,
        industries: tuple[str, ...],
        constraints: PortfolioConstraints,
    ) -> OptimizationResult:
        n = len(symbols)
        alpha = np.asarray(scores, dtype=np.float64)
        current = np.asarray(current_weights, dtype=np.float64)
        benchmark = np.asarray(benchmark_weights, dtype=np.float64)
        if n == 0 or alpha.shape != (n,) or current.shape != (n,) or benchmark.shape != (n,):
            raise ValueError("optimizer vector shapes are invalid")
        if len(industries) != n or not np.all(np.isfinite(alpha)):
            raise ValueError("optimizer industry/score data are invalid")

        weights = cp.Variable(n)
        objective = cp.Maximize(
            alpha @ weights - constraints.risk_aversion * cp.sum_squares(weights - benchmark)
        )
        rules: list[Any] = [
            weights >= 0,
            weights <= constraints.maximum_weight,
            cp.sum(weights) <= 1 - constraints.minimum_cash_weight,
            cp.norm1(weights - current) <= 2 * constraints.maximum_turnover,
        ]
        for industry in sorted(set(industries)):
            mask = np.asarray([value == industry for value in industries], dtype=np.float64)
            benchmark_industry = float(mask @ benchmark)
            active = mask @ weights - benchmark_industry
            rules.extend(
                [
                    active <= constraints.maximum_industry_deviation,
                    active >= -constraints.maximum_industry_deviation,
                ]
            )
        problem = cp.Problem(objective, rules)
        value = problem.solve(solver=cp.CLARABEL)
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
            raise ValueError(f"portfolio optimization failed: {problem.status}")
        resolved = np.maximum(np.asarray(weights.value, dtype=np.float64), 0)
        turnover = float(np.sum(np.abs(resolved - current)) / 2)
        return OptimizationResult(
            {symbol: float(weight) for symbol, weight in zip(symbols, resolved, strict=True)},
            float(1 - np.sum(resolved)),
            turnover,
            float(value),
        )
