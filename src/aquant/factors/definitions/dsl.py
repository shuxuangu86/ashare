import ast
import hashlib
import json
from dataclasses import dataclass
from typing import TypeAlias, cast

Scalar: TypeAlias = int | float
Argument: TypeAlias = "Expression | Scalar | str"

RAW_FIELDS = frozenset({"Open", "High", "Low", "Close", "Volume", "Amount", "FloatMV", "PE", "PB"})
ARITIES: dict[str, tuple[int, int]] = {
    "Ref": (2, 2),
    "Delta": (2, 2),
    "Mean": (2, 2),
    "Sum": (2, 2),
    "Std": (2, 2),
    "Min": (2, 2),
    "Max": (2, 2),
    "RankTS": (2, 2),
    "Corr": (3, 3),
    "Cov": (3, 3),
    "RankCS": (1, 1),
    "ZScore": (1, 1),
    "Winsorize": (1, 2),
    "Neutralize": (2, 2),
    "Add": (2, 2),
    "Sub": (2, 2),
    "Mul": (2, 2),
    "Div": (2, 2),
    "Log": (1, 1),
    "Abs": (1, 1),
    "Sign": (1, 1),
    "Power": (2, 2),
}
WINDOW_OPERATORS = frozenset({"Ref", "Delta", "Mean", "Sum", "Std", "Min", "Max", "RankTS"})


class FactorExpressionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Expression:
    operator: str
    arguments: tuple[Argument, ...] = ()

    @property
    def canonical(self) -> str:
        def encode(argument: Argument) -> object:
            if isinstance(argument, Expression):
                return {
                    "op": argument.operator,
                    "args": [encode(item) for item in argument.arguments],
                }
            return argument

        return json.dumps(encode(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def expression_hash(self) -> str:
        return hashlib.sha256(self.canonical.encode()).hexdigest()

    @property
    def complexity(self) -> int:
        return 1 + sum(
            argument.complexity if isinstance(argument, Expression) else 0
            for argument in self.arguments
        )

    @property
    def required_fields(self) -> tuple[str, ...]:
        fields: set[str] = set()

        def collect(expression: Expression) -> None:
            if expression.operator == "Field":
                fields.add(str(expression.arguments[0]))
            for argument in expression.arguments:
                if isinstance(argument, Expression):
                    collect(argument)

        collect(self)
        return tuple(sorted(fields))


def parse_expression(source: str, *, max_complexity: int = 64) -> Expression:
    if not source.strip():
        raise FactorExpressionError("factor expression must not be blank")
    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise FactorExpressionError("factor expression syntax is invalid") from exc
    expression = _convert(parsed.body)
    if expression.complexity > max_complexity:
        raise FactorExpressionError("factor expression exceeds complexity limit")
    return expression


def _convert(node: ast.expr) -> Expression:
    if isinstance(node, ast.Name):
        if node.id not in RAW_FIELDS:
            raise FactorExpressionError(f"field is not allowed: {node.id}")
        return Expression("Field", (node.id,))
    if isinstance(node, ast.Call):
        if node.keywords or not isinstance(node.func, ast.Name):
            raise FactorExpressionError(
                "only positional calls to whitelisted operators are allowed"
            )
        operator = node.func.id
        if operator not in ARITIES:
            raise FactorExpressionError(f"operator is not allowed: {operator}")
        arguments = tuple(_convert_argument(argument) for argument in node.args)
        minimum, maximum = ARITIES[operator]
        if not minimum <= len(arguments) <= maximum:
            raise FactorExpressionError(f"invalid arity for {operator}")
        if operator in WINDOW_OPERATORS:
            _positive_window(arguments[-1], operator)
        if operator in {"Corr", "Cov"}:
            _positive_window(arguments[-1], operator)
        return Expression(operator, arguments)
    raise FactorExpressionError(f"syntax is not allowed: {type(node).__name__}")


def _convert_argument(node: ast.expr) -> Argument:
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return cast(Scalar, node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) in {int, float}
    ):
        return -cast(Scalar, node.operand.value)
    return _convert(node)


def _positive_window(value: Argument, operator: str) -> None:
    if type(value) is not int or value <= 0:
        raise FactorExpressionError(f"{operator} window must be a positive integer")
