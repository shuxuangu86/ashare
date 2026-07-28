import ast
import hashlib
import json
from dataclasses import dataclass
from typing import TypeAlias, cast

Scalar: TypeAlias = int | float
Argument: TypeAlias = "Expression | Scalar | str"

RAW_FIELDS = frozenset(
    {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Amount",
        "VWAP",
        "TurnoverRate",
        "TotalMV",
        "FloatMV",
        "FreeFloatMV",
        "PE",
        "PB",
        "Revenue",
        "NetProfit",
        "OperatingCashFlow",
        "TotalAssets",
        "TotalEquity",
        "TotalLiabilities",
        "Industry",
    }
)
ARITIES: dict[str, tuple[int, int]] = {
    "Ref": (2, 2),
    "Delay": (2, 2),
    "Delta": (2, 2),
    "Return": (2, 2),
    "Mean": (2, 2),
    "Sum": (2, 2),
    "Std": (2, 2),
    "Min": (2, 2),
    "Max": (2, 2),
    "RankTS": (2, 2),
    "TsRank": (2, 2),
    "Corr": (3, 3),
    "Cov": (3, 3),
    "RankCS": (1, 1),
    "Rank": (1, 1),
    "ZScore": (1, 1),
    "Winsorize": (1, 2),
    "Neutralize": (2, 2),
    "DecayLinear": (2, 2),
    "SignedPower": (2, 2),
    "Scale": (1, 2),
    "ArgMax": (2, 2),
    "ArgMin": (2, 2),
    "Where": (3, 3),
    "Greater": (2, 2),
    "GreaterEqual": (2, 2),
    "Less": (2, 2),
    "LessEqual": (2, 2),
    "Equal": (2, 2),
    "NotEqual": (2, 2),
    "Add": (2, 2),
    "Sub": (2, 2),
    "Mul": (2, 2),
    "Div": (2, 2),
    "Log": (1, 1),
    "Abs": (1, 1),
    "Sign": (1, 1),
    "Power": (2, 2),
}
WINDOW_OPERATORS = frozenset(
    {
        "Ref",
        "Delay",
        "Delta",
        "Return",
        "Mean",
        "Sum",
        "Std",
        "Min",
        "Max",
        "RankTS",
        "TsRank",
        "DecayLinear",
        "ArgMax",
        "ArgMin",
    }
)
PAIR_WINDOW_OPERATORS = frozenset({"Corr", "Cov"})
BINARIES = {
    ast.Add: "Add",
    ast.Sub: "Sub",
    ast.Mult: "Mul",
    ast.Div: "Div",
    ast.Pow: "Power",
}
COMPARISONS = {
    ast.Gt: "Greater",
    ast.GtE: "GreaterEqual",
    ast.Lt: "Less",
    ast.LtE: "LessEqual",
    ast.Eq: "Equal",
    ast.NotEq: "NotEqual",
}


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
    def depth(self) -> int:
        child_depths = [
            argument.depth for argument in self.arguments if isinstance(argument, Expression)
        ]
        return 1 + max(child_depths, default=0)

    @property
    def required_fields(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    str(expression.arguments[0])
                    for expression in self.walk()
                    if expression.operator == "Field"
                }
            )
        )

    @property
    def history_requirement(self) -> int:
        children = [
            argument.history_requirement
            for argument in self.arguments
            if isinstance(argument, Expression)
        ]
        base = max(children, default=0)
        if self.operator in {"Ref", "Delay", "Delta", "Return"}:
            return base + _window_value(self)
        if self.operator in WINDOW_OPERATORS | PAIR_WINDOW_OPERATORS:
            return base + _window_value(self) - 1
        return base

    @property
    def complexity_score(self) -> float:
        walked = self.walk()
        windows = sum(
            expression.operator in WINDOW_OPERATORS | PAIR_WINDOW_OPERATORS for expression in walked
        )
        branches = sum(expression.operator == "Where" for expression in walked)
        neutralizations = sum(expression.operator == "Neutralize" for expression in walked)
        return float(
            self.complexity
            + self.depth
            + len(self.required_fields)
            + windows
            + 2 * branches
            + 2 * neutralizations
        )

    def walk(self) -> tuple["Expression", ...]:
        descendants = [self]
        for argument in self.arguments:
            if isinstance(argument, Expression):
                descendants.extend(argument.walk())
        return tuple(descendants)


def parse_expression(
    source: str,
    *,
    max_complexity: int = 40,
    max_depth: int = 6,
    max_fields: int = 5,
) -> Expression:
    if not source.strip():
        raise FactorExpressionError("factor expression must not be blank")
    try:
        parsed = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        raise FactorExpressionError("factor expression syntax is invalid") from exc
    expression = _convert(parsed.body)
    if expression.complexity > max_complexity:
        raise FactorExpressionError("factor expression exceeds complexity limit")
    if expression.depth > max_depth:
        raise FactorExpressionError("factor expression exceeds depth limit")
    if len(expression.required_fields) > max_fields:
        raise FactorExpressionError("factor expression exceeds input-field limit")
    return expression


def _convert(node: ast.expr) -> Expression:
    if isinstance(node, ast.Name):
        if node.id not in RAW_FIELDS:
            raise FactorExpressionError(f"field is not allowed: {node.id}")
        return Expression("Field", (node.id,))
    if isinstance(node, ast.BinOp) and type(node.op) in BINARIES:
        return Expression(
            BINARIES[type(node.op)],
            (_convert_argument(node.left), _convert_argument(node.right)),
        )
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise FactorExpressionError("chained comparisons are not allowed")
        operator = COMPARISONS.get(type(node.ops[0]))
        if operator is None:
            raise FactorExpressionError("comparison operator is not allowed")
        return Expression(
            operator,
            (_convert_argument(node.left), _convert_argument(node.comparators[0])),
        )
    if isinstance(node, ast.IfExp):
        return Expression(
            "Where",
            (
                _convert_argument(node.test),
                _convert_argument(node.body),
                _convert_argument(node.orelse),
            ),
        )
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
        if operator in WINDOW_OPERATORS | PAIR_WINDOW_OPERATORS:
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


def _window_value(expression: Expression) -> int:
    value = expression.arguments[-1]
    if type(value) is not int:
        raise FactorExpressionError(f"{expression.operator} window must be an integer")
    return value
