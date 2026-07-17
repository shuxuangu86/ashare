from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aquant.factors.evaluation import FactorAnalyzer, FactorEvaluation
from aquant.factors.mining.generator import FactorCandidate

Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int

    def __post_init__(self) -> None:
        if not (
            0 <= self.train_start < self.train_end <= self.validation_start < self.validation_end
        ):
            raise ValueError("walk-forward split must be ordered and non-overlapping")


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate: FactorCandidate
    train: FactorEvaluation
    validation: FactorEvaluation
    score: float


class CandidateSelector:
    def evaluate(
        self,
        candidate: FactorCandidate,
        factor_values: Array,
        forward_returns: Array,
        split: WalkForwardSplit,
    ) -> CandidateEvaluation:
        analyzer = FactorAnalyzer()
        train = analyzer.evaluate(
            factor_values[split.train_start : split.train_end],
            forward_returns[split.train_start : split.train_end],
        )
        validation = analyzer.evaluate(
            factor_values[split.validation_start : split.validation_end],
            forward_returns[split.validation_start : split.validation_end],
        )
        ir = validation.rank_ic_ir if np.isfinite(validation.rank_ic_ir) else 0.0
        monotonicity = validation.monotonicity if np.isfinite(validation.monotonicity) else 0.0
        turnover = validation.turnover if np.isfinite(validation.turnover) else 1.0
        score = ir + monotonicity - turnover - candidate.expression.complexity * 0.01
        return CandidateEvaluation(candidate, train, validation, float(score))

    def correlation_filter(
        self,
        candidates: tuple[tuple[CandidateEvaluation, Array], ...],
        *,
        maximum_absolute_correlation: float = 0.8,
    ) -> tuple[CandidateEvaluation, ...]:
        if not 0 <= maximum_absolute_correlation <= 1:
            raise ValueError("correlation threshold must be in [0, 1]")
        accepted: list[tuple[CandidateEvaluation, Array]] = []
        for evaluation, raw_values in sorted(
            candidates, key=lambda item: item[0].score, reverse=True
        ):
            values = np.asarray(raw_values, dtype=np.float64).ravel()
            if all(
                self._absolute_correlation(values, existing) <= maximum_absolute_correlation
                for _, existing in accepted
            ):
                accepted.append((evaluation, values))
        return tuple(evaluation for evaluation, _ in accepted)

    @staticmethod
    def _absolute_correlation(left: Array, right: Array) -> float:
        valid = np.isfinite(left) & np.isfinite(right)
        if np.count_nonzero(valid) < 2:
            return 1.0
        x = left[valid]
        y = right[valid]
        if np.std(x) == 0 or np.std(y) == 0:
            return 1.0
        return abs(float(np.corrcoef(x, y)[0, 1]))


def benjamini_hochberg(p_values: tuple[float, ...], *, alpha: float = 0.05) -> tuple[bool, ...]:
    if not 0 < alpha < 1 or any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("p-values and alpha are invalid")
    count = len(p_values)
    accepted = [False] * count
    threshold_rank = 0
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    for rank, (_, value) in enumerate(ordered, start=1):
        if value <= alpha * rank / count:
            threshold_rank = rank
    for original_index, _ in ordered[:threshold_rank]:
        accepted[original_index] = True
    return tuple(accepted)
