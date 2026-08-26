from dataclasses import dataclass
from datetime import datetime

from aquant.factors.definitions.dsl import parse_expression
from aquant.factors.generation.budgets import enforce_budget
from aquant.factors.spec import FactorLayer, FactorSpec, FactorStatus, SourceType

SPARSE_WINDOWS = (5, 10, 20, 40, 60, 120, 250)


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    batch_id: str
    family_budget: str
    candidates: tuple[FactorSpec, ...]
    attempted: int
    failures: tuple[str, ...]


def generate_window_variants(
    parent: FactorSpec,
    *,
    template: str,
    windows: tuple[int, ...] = SPARSE_WINDOWS,
    batch_id: str,
    family_budget: str,
    created_at: datetime,
) -> CandidateBatch:
    if not batch_id.strip() or len(windows) != len(set(windows)) or any(x <= 0 for x in windows):
        raise ValueError("candidate batch/windows must be valid and unique")
    enforce_budget(family_budget, len(windows))
    candidates: list[FactorSpec] = []
    failures: list[str] = []
    hashes: set[str] = set()
    for window in windows:
        try:
            expression = template.format(window=window)
            parsed = parse_expression(expression)
            candidate = FactorSpec(
                **{
                    **parent.model_dump(),
                    "factor_id": f"{parent.factor_id}__window_{window}",
                    "name": f"{parent.name} window {window}",
                    "version": "1.0.0",
                    "status": FactorStatus.CANDIDATE,
                    "layer": FactorLayer.L2B,
                    "expression": expression,
                    "implementation": None,
                    "parent_factor_ids": (parent.factor_id,),
                    "variant_dimension": "window",
                    "input_fields": parsed.required_fields,
                    "required_history": parsed.history_requirement,
                    "parameters": {
                        **parent.parameters,
                        "window": window,
                        "generation_batch": batch_id,
                    },
                    "source_type": SourceType.GENERATED,
                    "source_reference": batch_id,
                    "complexity_score": parsed.complexity_score,
                    "expression_hash": "",
                    "created_at": created_at,
                    "updated_at": created_at,
                }
            )
            if candidate.expression_hash not in hashes:
                hashes.add(candidate.expression_hash)
                candidates.append(candidate)
        except ValueError as exc:
            failures.append(f"window={window}: {exc}")
    return CandidateBatch(batch_id, family_budget, tuple(candidates), len(windows), tuple(failures))
