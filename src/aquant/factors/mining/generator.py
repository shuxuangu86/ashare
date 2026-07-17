import random
from dataclasses import dataclass

from aquant.factors.definitions import Expression, parse_expression


@dataclass(frozen=True, slots=True)
class FactorCandidate:
    expression: Expression
    origin: str

    @property
    def candidate_id(self) -> str:
        return self.expression.expression_hash


class CandidateGenerator:
    """Generates only parseable, whitelisted DSL expressions."""

    def enumerate_templates(
        self,
        *,
        fields: tuple[str, ...],
        windows: tuple[int, ...],
    ) -> tuple[FactorCandidate, ...]:
        candidates: list[FactorCandidate] = []
        for field in fields:
            for window in windows:
                sources = (
                    f"RankCS(Delta({field}, {window}))",
                    f"RankCS(Div({field}, Mean({field}, {window})))",
                    f"RankCS(Mul(Std({field}, {window}), -1))",
                )
                candidates.extend(
                    FactorCandidate(parse_expression(source), "template") for source in sources
                )
        return self._deduplicate(candidates)

    def random_search(
        self,
        *,
        fields: tuple[str, ...],
        windows: tuple[int, ...],
        count: int,
        seed: int,
    ) -> tuple[FactorCandidate, ...]:
        if count < 0 or not fields or not windows:
            raise ValueError("random search requires fields/windows and non-negative count")
        generator = random.Random(seed)
        operators = ("Mean", "Std", "Delta", "RankTS")
        arithmetic = ("Add", "Sub", "Mul", "Div")
        candidates: list[FactorCandidate] = []
        attempts = 0
        while len(candidates) < count and attempts < max(count * 20, 1):
            attempts += 1
            left = generator.choice(fields)
            right = generator.choice(fields)
            window = generator.choice(windows)
            operation = generator.choice(operators)
            combine = generator.choice(arithmetic)
            source = f"RankCS({combine}({operation}({left}, {window}), {right}))"
            candidates.append(FactorCandidate(parse_expression(source), "random"))
            candidates = list(self._deduplicate(candidates))
        return tuple(candidates[:count])

    @staticmethod
    def _deduplicate(candidates: list[FactorCandidate]) -> tuple[FactorCandidate, ...]:
        unique: dict[str, FactorCandidate] = {}
        for candidate in candidates:
            unique.setdefault(candidate.candidate_id, candidate)
        return tuple(unique.values())
