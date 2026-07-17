from dataclasses import dataclass
from decimal import Decimal

from aquant.domain.identifiers import Symbol
from aquant.execution.broker_api import BrokerAccountSnapshot


@dataclass(frozen=True, slots=True)
class ReconciliationDifference:
    field: str
    symbol: Symbol | None
    local_value: str
    broker_value: str


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    matched: bool
    differences: tuple[ReconciliationDifference, ...]
    broker_snapshot: BrokerAccountSnapshot


def reconcile_accounts(
    local: BrokerAccountSnapshot,
    broker: BrokerAccountSnapshot,
    *,
    cash_tolerance: Decimal = Decimal("0.01"),
) -> ReconciliationReport:
    differences: list[ReconciliationDifference] = []
    if abs(local.cash - broker.cash) > cash_tolerance:
        differences.append(
            ReconciliationDifference("cash", None, str(local.cash), str(broker.cash))
        )
    for symbol in sorted(set(local.positions) | set(broker.positions)):
        local_quantity = local.positions.get(symbol, 0)
        broker_quantity = broker.positions.get(symbol, 0)
        if local_quantity != broker_quantity:
            differences.append(
                ReconciliationDifference(
                    "position", symbol, str(local_quantity), str(broker_quantity)
                )
            )
    return ReconciliationReport(not differences, tuple(differences), broker)
