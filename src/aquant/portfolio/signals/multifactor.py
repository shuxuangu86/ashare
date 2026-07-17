from dataclasses import dataclass

import numpy as np

from aquant.domain.identifiers import Symbol


@dataclass(frozen=True, slots=True)
class ScoredSecurity:
    symbol: Symbol
    score: float


class MultiFactorScorer:
    def score(
        self,
        factor_values: dict[str, dict[Symbol, float]],
        weights: dict[str, float],
    ) -> tuple[ScoredSecurity, ...]:
        if not factor_values or set(factor_values) != set(weights):
            raise ValueError("factor values and weights must have identical non-empty keys")
        if (
            not all(np.isfinite(weight) for weight in weights.values())
            or sum(abs(weight) for weight in weights.values()) == 0
        ):
            raise ValueError("factor weights must be finite and non-zero")
        common = set.intersection(*(set(values) for values in factor_values.values()))
        if not common:
            raise ValueError("no securities have complete factor coverage")
        ordered = tuple(sorted(common))
        composite = np.zeros(len(ordered), dtype=np.float64)
        scale = sum(abs(value) for value in weights.values())
        for name, values in factor_values.items():
            sample = np.asarray([values[symbol] for symbol in ordered], dtype=np.float64)
            if not np.all(np.isfinite(sample)):
                raise ValueError(f"factor contains non-finite values: {name}")
            deviation = float(np.std(sample))
            standardized = (
                np.zeros_like(sample) if deviation == 0 else (sample - np.mean(sample)) / deviation
            )
            composite += weights[name] / scale * standardized
        return tuple(
            sorted(
                (
                    ScoredSecurity(symbol, float(score))
                    for symbol, score in zip(ordered, composite, strict=True)
                ),
                key=lambda item: (-item.score, item.symbol),
            )
        )
