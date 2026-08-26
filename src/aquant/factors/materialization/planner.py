from aquant.factors.atomic.models import AtomicFactor


def plan_dependencies(factors: tuple[AtomicFactor, ...]) -> tuple[AtomicFactor, ...]:
    by_id = {factor.spec.factor_id: factor for factor in factors}
    if len(by_id) != len(factors):
        raise ValueError("materialization request contains duplicate factor ids")
    ordered: list[AtomicFactor] = []
    complete: set[str] = set()

    def visit(factor: AtomicFactor, active: set[str]) -> None:
        factor_id = factor.spec.factor_id
        if factor_id in complete:
            return
        if factor_id in active:
            raise ValueError("factor dependency graph contains a cycle")
        active.add(factor_id)
        for parent_id in factor.spec.parent_factor_ids:
            try:
                parent = by_id[parent_id]
            except KeyError as exc:
                raise ValueError(f"materialization dependency is missing: {parent_id}") from exc
            visit(parent, active)
        active.remove(factor_id)
        complete.add(factor_id)
        ordered.append(factor)

    for factor in sorted(factors, key=lambda item: item.spec.factor_id):
        visit(factor, set())
    return tuple(ordered)
