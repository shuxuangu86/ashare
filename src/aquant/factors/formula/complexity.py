from dataclasses import dataclass

from aquant.factors.definitions.dsl import Expression


@dataclass(frozen=True, slots=True)
class ExpressionMetrics:
    node_count: int
    depth: int
    input_fields: tuple[str, ...]
    history_requirement: int
    window_count: int
    condition_count: int
    neutralization_count: int
    complexity_score: float


def analyze_expression(expression: Expression) -> ExpressionMetrics:
    walked = expression.walk()
    return ExpressionMetrics(
        node_count=expression.complexity,
        depth=expression.depth,
        input_fields=expression.required_fields,
        history_requirement=expression.history_requirement,
        window_count=sum(
            item.operator
            in {"Ref", "Delay", "Delta", "Return", "Mean", "Std", "Corr", "Cov", "TsRank"}
            for item in walked
        ),
        condition_count=sum(item.operator == "Where" for item in walked),
        neutralization_count=sum(item.operator == "Neutralize" for item in walked),
        complexity_score=expression.complexity_score,
    )
