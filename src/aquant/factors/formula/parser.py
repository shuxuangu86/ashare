from aquant.factors.definitions.dsl import Expression, parse_expression
from aquant.factors.formula.validator import DEFAULT_LIMITS, ExpressionLimits


def parse_formula(
    source: str,
    *,
    limits: ExpressionLimits = DEFAULT_LIMITS,
) -> Expression:
    return parse_expression(
        source,
        max_complexity=limits.max_nodes,
        max_depth=limits.max_depth,
        max_fields=limits.max_fields,
    )
