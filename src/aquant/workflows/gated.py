from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StageResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    completed: bool
    stages: tuple[StageResult, ...]


class FailClosedWorkflow:
    def __init__(self, stages: tuple[tuple[str, Callable[[], StageResult]], ...]) -> None:
        self._stages = stages

    def run(self) -> WorkflowResult:
        results: list[StageResult] = []
        for expected_name, stage in self._stages:
            result = stage()
            if result.name != expected_name:
                raise ValueError("workflow stage returned an unexpected identity")
            results.append(result)
            if not result.passed:
                return WorkflowResult(False, tuple(results))
        return WorkflowResult(True, tuple(results))
