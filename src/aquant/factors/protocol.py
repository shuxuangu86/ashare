from typing import Protocol

from aquant.factors.spec import FactorSpec
from aquant.factors.types import FactorContext, FactorResult


class Factor(Protocol):
    spec: FactorSpec

    def compute(self, context: FactorContext) -> FactorResult:
        """Compute only through context.input_loader using PIT-visible inputs."""
        ...
