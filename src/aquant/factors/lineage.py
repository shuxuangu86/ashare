from dataclasses import dataclass

from aquant.factors.spec import FactorSpec


@dataclass(frozen=True, slots=True)
class FactorLineage:
    factor_id: str
    factor_version: str
    parent_factor_ids: tuple[str, ...]
    variant_dimension: str | None
    source_reference: str
    expression_hash: str

    @classmethod
    def from_spec(cls, spec: FactorSpec) -> "FactorLineage":
        return cls(
            factor_id=spec.factor_id,
            factor_version=spec.version,
            parent_factor_ids=spec.parent_factor_ids,
            variant_dimension=spec.variant_dimension,
            source_reference=spec.source_reference,
            expression_hash=spec.expression_hash,
        )


def validate_lineage(specs: tuple[FactorSpec, ...]) -> None:
    known = {spec.factor_id for spec in specs}
    graph = {spec.factor_id: set(spec.parent_factor_ids) for spec in specs}
    missing = {parent for parents in graph.values() for parent in parents if parent not in known}
    if missing:
        raise ValueError(f"lineage references unknown parents: {sorted(missing)}")

    def visit(node: str, active: set[str], complete: set[str]) -> None:
        if node in active:
            raise ValueError("factor lineage contains a cycle")
        if node in complete:
            return
        active.add(node)
        for parent in graph[node]:
            visit(parent, active, complete)
        active.remove(node)
        complete.add(node)

    complete: set[str] = set()
    for factor_id in graph:
        visit(factor_id, set(), complete)
