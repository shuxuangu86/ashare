from collections.abc import Callable
from dataclasses import dataclass

from aquant.execution.broker_api import BrokerAdapter, BrokerOrderRequest
from aquant.execution.kill_switch import KillSwitch
from aquant.execution.order_manager import ExecutionOrderStore, ExecutionRecordStatus
from aquant.portfolio.risk.pretrade import PreTradeContext, PreTradeRiskEngine


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    submitted: int
    rejected: int
    unknown: int


class ExecutionAgent:
    """Windows-safe execution loop: approved orders, risk gate, broker, persistent status."""

    def __init__(
        self,
        *,
        store: ExecutionOrderStore,
        broker: BrokerAdapter,
        risk_engine: PreTradeRiskEngine,
        context_provider: Callable[[BrokerOrderRequest], PreTradeContext],
        kill_switch: KillSwitch,
    ) -> None:
        self._store = store
        self._broker = broker
        self._risk_engine = risk_engine
        self._context_provider = context_provider
        self._kill_switch = kill_switch

    def run_once(self) -> AgentRunResult:
        self._kill_switch.require_inactive()
        submitted = rejected = unknown = 0
        for record in self._store.pending():
            request = record.request
            decision = self._risk_engine.check(request, self._context_provider(request))
            if not decision.approved:
                self._store.mark_failed(request, ",".join(decision.reasons))
                rejected += 1
                continue
            try:
                order = self._broker.submit_order(request)
            except Exception as exc:
                self._store.mark_unknown(request, str(exc))
                unknown += 1
                continue
            self._store.mark_submitted(request, order)
            submitted += 1
        return AgentRunResult(submitted, rejected, unknown)

    def unresolved_count(self) -> int:
        return sum(record.status is ExecutionRecordStatus.UNKNOWN for record in self._store.records)
