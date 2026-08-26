from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aquant.factors.operators.cross_sectional import cs_rank

Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FactorEvaluation:
    ic_by_period: tuple[float, ...]
    rank_ic_by_period: tuple[float, ...]
    ic_mean: float
    rank_ic_mean: float
    rank_ic_std: float
    rank_ic_ir: float
    quantile_returns: tuple[float, ...]
    long_short_return: float
    monotonicity: float
    turnover: float
    coverage: float


class FactorAnalyzer:
    def evaluate(
        self,
        factor_values: Array,
        forward_returns: Array,
        *,
        quantiles: int = 5,
    ) -> FactorEvaluation:
        factors, returns = self._pair(factor_values, forward_returns)
        if quantiles < 2:
            raise ValueError("quantiles must be at least two")
        ic: list[float] = []
        rank_ic: list[float] = []
        grouped: list[list[float]] = [[] for _ in range(quantiles)]
        previous_assignment: Array | None = None
        turnover_sum = 0.0
        turnover_periods = 0
        for factor_row, return_row in zip(factors, returns, strict=True):
            valid = np.isfinite(factor_row) & np.isfinite(return_row)
            if np.count_nonzero(valid) < 2:
                row_assignment = np.full(factor_row.shape, np.nan)
                previous_assignment = row_assignment
                continue
            x = factor_row[valid]
            y = return_row[valid]
            ic.append(self._correlation(x, y))
            rank_ic.append(self._correlation(self._rank(x), self._rank(y)))
            row_assignment = np.full(factor_row.shape, np.nan)
            ranks = self._rank(x)
            groups = np.minimum((ranks * quantiles / len(x)).astype(int), quantiles - 1)
            row_assignment[valid] = groups
            if previous_assignment is not None:
                turnover_valid = np.isfinite(previous_assignment) & np.isfinite(row_assignment)
                if np.any(turnover_valid):
                    turnover_sum += float(
                        np.mean(
                            previous_assignment[turnover_valid] != row_assignment[turnover_valid]
                        )
                    )
                    turnover_periods += 1
            previous_assignment = row_assignment
            for group in range(quantiles):
                group_returns = y[groups == group]
                if group_returns.size:
                    grouped[group].append(float(np.mean(group_returns)))
        quantile_returns = tuple(
            float(np.mean(values)) if values else float("nan") for values in grouped
        )
        finite_ic = np.asarray(rank_ic, dtype=np.float64)
        rank_mean = float(np.nanmean(finite_ic)) if finite_ic.size else float("nan")
        rank_std = float(np.nanstd(finite_ic)) if finite_ic.size else float("nan")
        return FactorEvaluation(
            ic_by_period=tuple(ic),
            rank_ic_by_period=tuple(rank_ic),
            ic_mean=float(np.nanmean(ic)) if ic else float("nan"),
            rank_ic_mean=rank_mean,
            rank_ic_std=rank_std,
            rank_ic_ir=rank_mean / rank_std if rank_std > 0 else float("nan"),
            quantile_returns=quantile_returns,
            long_short_return=quantile_returns[-1] - quantile_returns[0],
            monotonicity=self._correlation(
                np.arange(quantiles, dtype=np.float64), np.asarray(quantile_returns)
            ),
            turnover=(turnover_sum / turnover_periods if turnover_periods else float("nan")),
            coverage=float(np.count_nonzero(np.isfinite(factors)) / factors.size),
        )

    def decay_rank_ic(
        self, factor_values: Array, forward_returns_by_horizon: dict[int, Array]
    ) -> tuple[tuple[int, float], ...]:
        if not forward_returns_by_horizon:
            raise ValueError("factor decay requires at least one horizon")
        result: list[tuple[int, float]] = []
        for horizon, returns in sorted(forward_returns_by_horizon.items()):
            if horizon <= 0:
                raise ValueError("factor decay horizons must be positive")
            evaluation = self.evaluate(factor_values, returns)
            result.append((horizon, evaluation.rank_ic_mean))
        return tuple(result)

    @staticmethod
    def _pair(left: Array, right: Array) -> tuple[Array, Array]:
        x = np.asarray(left, dtype=np.float64)
        y = np.asarray(right, dtype=np.float64)
        if x.ndim != 2 or x.shape != y.shape or not x.size:
            raise ValueError("factor and return arrays must be non-empty with equal 2D shape")
        return x, y

    @staticmethod
    def _rank(values: Array) -> Array:
        # Quantile assignment below expects zero-based ranks. Subtracting one
        # preserves average ranks for ties without shifting group boundaries.
        return np.asarray(cs_rank(values), dtype=np.float64) - 1.0

    @staticmethod
    def _correlation(left: Array, right: Array) -> float:
        valid = np.isfinite(left) & np.isfinite(right)
        if np.count_nonzero(valid) < 2:
            return float("nan")
        x = left[valid]
        y = right[valid]
        if np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])
