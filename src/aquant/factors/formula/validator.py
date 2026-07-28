from dataclasses import dataclass

from aquant.factors.definitions.dsl import Expression, FactorExpressionError


@dataclass(frozen=True, slots=True)
class ExpressionLimits:
    max_depth: int = 6
    max_nodes: int = 40
    max_fields: int = 5

    def __post_init__(self) -> None:
        if min(self.max_depth, self.max_nodes, self.max_fields) <= 0:
            raise ValueError("expression limits must be positive")


DEFAULT_LIMITS = ExpressionLimits()


def validate_expression(
    expression: Expression,
    *,
    limits: ExpressionLimits = DEFAULT_LIMITS,
) -> None:
    if expression.depth > limits.max_depth:
        raise FactorExpressionError("factor expression exceeds depth limit")
    if expression.complexity > limits.max_nodes:
        raise FactorExpressionError("factor expression exceeds complexity limit")
    if len(expression.required_fields) > limits.max_fields:
        raise FactorExpressionError("factor expression exceeds input-field limit")
    forbidden = {"Lead", "Future", "CenteredMean"}
    found = forbidden.intersection(item.operator for item in expression.walk())
    if found:
        raise FactorExpressionError(f"future operators are forbidden: {sorted(found)}")
