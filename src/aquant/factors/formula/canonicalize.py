from aquant.factors.definitions.dsl import parse_expression


def canonicalize_expression(source: str) -> str:
    return parse_expression(source).canonical


def expression_hash(source: str) -> str:
    return parse_expression(source).expression_hash
