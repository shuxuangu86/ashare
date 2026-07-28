from aquant.factors.exceptions import FactorLifecycleError
from aquant.factors.spec import FactorStatus

_TRANSITIONS: dict[FactorStatus, frozenset[FactorStatus]] = {
    FactorStatus.CANDIDATE: frozenset(
        {FactorStatus.DRAFT, FactorStatus.UNAVAILABLE, FactorStatus.ARCHIVED}
    ),
    FactorStatus.DRAFT: frozenset(
        {FactorStatus.COMPUTED, FactorStatus.UNAVAILABLE, FactorStatus.ARCHIVED}
    ),
    FactorStatus.COMPUTED: frozenset(
        {
            FactorStatus.VALIDATED,
            FactorStatus.REDUNDANT,
            FactorStatus.CONDITIONAL,
            FactorStatus.RISK_ONLY,
            FactorStatus.ARCHIVED,
            FactorStatus.DEPRECATED,
        }
    ),
    FactorStatus.VALIDATED: frozenset(
        {
            FactorStatus.APPROVED,
            FactorStatus.REDUNDANT,
            FactorStatus.CONDITIONAL,
            FactorStatus.RISK_ONLY,
            FactorStatus.ARCHIVED,
            FactorStatus.DEPRECATED,
        }
    ),
    FactorStatus.APPROVED: frozenset(
        {
            FactorStatus.PAPER_TRADING,
            FactorStatus.PRODUCTION,
            FactorStatus.DECAYED,
            FactorStatus.DEPRECATED,
        }
    ),
    FactorStatus.PAPER_TRADING: frozenset(
        {FactorStatus.PRODUCTION, FactorStatus.DECAYED, FactorStatus.DEPRECATED}
    ),
    FactorStatus.PRODUCTION: frozenset({FactorStatus.DECAYED, FactorStatus.DEPRECATED}),
    FactorStatus.DECAYED: frozenset(
        {FactorStatus.VALIDATED, FactorStatus.ARCHIVED, FactorStatus.DEPRECATED}
    ),
    FactorStatus.REDUNDANT: frozenset({FactorStatus.ARCHIVED, FactorStatus.DEPRECATED}),
    FactorStatus.CONDITIONAL: frozenset(
        {FactorStatus.VALIDATED, FactorStatus.ARCHIVED, FactorStatus.DEPRECATED}
    ),
    FactorStatus.RISK_ONLY: frozenset({FactorStatus.ARCHIVED, FactorStatus.DEPRECATED}),
    FactorStatus.UNAVAILABLE: frozenset({FactorStatus.DRAFT, FactorStatus.ARCHIVED}),
    FactorStatus.ARCHIVED: frozenset(),
    FactorStatus.DEPRECATED: frozenset(),
}


def require_transition(current: FactorStatus, target: FactorStatus) -> None:
    if target not in _TRANSITIONS[current]:
        raise FactorLifecycleError(f"invalid factor lifecycle transition: {current} -> {target}")


def production_eligible(status: FactorStatus) -> bool:
    return status in {FactorStatus.APPROVED, FactorStatus.PAPER_TRADING, FactorStatus.PRODUCTION}
