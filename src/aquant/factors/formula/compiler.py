from dataclasses import dataclass

from aquant.factors.definitions.dsl import Expression
from aquant.factors.operators.evaluator import Array, ExpressionEvaluator, FactorPanel


@dataclass(frozen=True, slots=True)
class FormulaCompiler:
    """Compile a validated AST to the deterministic array evaluator; never uses eval."""

    expression: Expression

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.expression.required_fields

    @property
    def required_history(self) -> int:
        return self.expression.history_requirement

    def evaluate(self, panel: FactorPanel) -> Array:
        return ExpressionEvaluator().evaluate(self.expression, panel)
