import hashlib
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import LimitStatus, SecurityStatus
from aquant.execution import (
    BrokerAccountSnapshot,
    BrokerFill,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderStatus,
    ExecutionAgent,
    ExecutionOrderStore,
    ExecutionRecordStatus,
    KillSwitch,
    PaperBroker,
    PaperBrokerState,
    QmtAdapterState,
    QmtBrokerAdapter,
    reconcile_accounts,
)
from aquant.portfolio import (
    IntradayRiskMonitor,
    PreTradeContext,
    PreTradeRiskEngine,
)

SYMBOL = Symbol.parse("600000.XSHG")
NOW = datetime(2026, 7, 17, 1, 30, tzinfo=UTC)


def _request(side: Side = Side.BUY, quantity: int = 100) -> BrokerOrderRequest:
    key = hashlib.sha256(f"{side}:{quantity}".encode()).hexdigest()
    return BrokerOrderRequest("client-1", key, SYMBOL, side, quantity)


def _account(cash: str = "100000", quantity: int = 1000) -> BrokerAccountSnapshot:
    return BrokerAccountSnapshot(NOW, Decimal(cash), {SYMBOL: quantity})


def _context(**overrides: object) -> PreTradeContext:
    values: dict[str, object] = {
        "now": NOW,
        "market_data_at": NOW,
        "price": Decimal("10"),
        "daily_volume": Decimal("100000"),
        "security_status": SecurityStatus(
            SYMBOL, date(2026, 7, 17), False, False, LimitStatus.NONE
        ),
        "account": _account(),
        "data_release_published": True,
    }
    values.update(overrides)
    return PreTradeContext(**values)  # type: ignore[arg-type]


def test_pretrade_risk_approves_healthy_order_and_reports_all_failures() -> None:
    engine = PreTradeRiskEngine()
    assert engine.check(_request(), _context()).approved
    bad = engine.check(
        _request(quantity=20000),
        _context(
            market_data_at=NOW - timedelta(minutes=2),
            security_status=SecurityStatus(
                SYMBOL, date(2026, 7, 17), True, True, LimitStatus.UNKNOWN
            ),
            account=_account(cash="1", quantity=0),
            data_release_published=False,
        ),
    )
    assert not bad.approved
    assert {
        "DATA_RELEASE_UNPUBLISHED",
        "MARKET_DATA_STALE",
        "SECURITY_SUSPENDED",
        "SECURITY_ST",
        "ORDER_NOTIONAL_LIMIT",
        "VOLUME_PARTICIPATION_LIMIT",
        "INSUFFICIENT_CASH",
    }.issubset(bad.reasons)
    sell = PreTradeRiskEngine().check(_request(Side.SELL, 2000), _context())
    assert "INSUFFICIENT_POSITION" in sell.reasons


def test_execution_agent_submits_only_approved_orders_and_is_restart_safe() -> None:
    request = _request()
    store = ExecutionOrderStore()
    store.approve(request)
    state = PaperBrokerState(Decimal("100000"))
    broker = PaperBroker(state)
    broker.set_price(SYMBOL, Decimal("10"))
    kill = KillSwitch()
    agent = ExecutionAgent(
        store=store,
        broker=broker,
        risk_engine=PreTradeRiskEngine(),
        context_provider=lambda _: _context(account=broker.query_account()),
        kill_switch=kill,
    )
    assert agent.run_once().submitted == 1
    assert agent.run_once().submitted == 0
    assert len(broker.query_fills()) == 1
    assert store.records[0].status is ExecutionRecordStatus.SUBMITTED


def test_execution_agent_rejects_risk_and_marks_transport_unknown() -> None:
    request = _request()
    rejected_store = ExecutionOrderStore()
    rejected_store.approve(request)
    broker = PaperBroker(PaperBrokerState(Decimal("100000")))
    rejected = ExecutionAgent(
        store=rejected_store,
        broker=broker,
        risk_engine=PreTradeRiskEngine(),
        context_provider=lambda _: _context(data_release_published=False),
        kill_switch=KillSwitch(),
    ).run_once()
    assert rejected.rejected == 1

    class BrokenBroker(PaperBroker):
        def submit_order(self, request: BrokerOrderRequest) -> BrokerOrder:
            raise ConnectionError("network down")

    unknown_store = ExecutionOrderStore()
    unknown_store.approve(request)
    agent = ExecutionAgent(
        store=unknown_store,
        broker=BrokenBroker(PaperBrokerState(Decimal("100000"))),
        risk_engine=PreTradeRiskEngine(),
        context_provider=lambda _: _context(),
        kill_switch=KillSwitch(),
    )
    assert agent.run_once().unknown == 1
    assert agent.unresolved_count() == 1


class FakeQmtGateway:
    def __init__(self) -> None:
        self.submissions = 0
        self.order: BrokerOrder | None = None

    def submit_order(self, request: BrokerOrderRequest) -> BrokerOrder:
        self.submissions += 1
        self.order = BrokerOrder("QMT-1", request, BrokerOrderStatus.SUBMITTED, NOW)
        return self.order

    def query_order(self, broker_order_id: str) -> BrokerOrder:
        assert self.order is not None and broker_order_id == self.order.broker_order_id
        return self.order

    def query_account(self) -> BrokerAccountSnapshot:
        return _account()

    def query_fills(self) -> tuple[BrokerFill, ...]:
        return ()

    def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        order = self.query_order(broker_order_id)
        self.order = BrokerOrder(
            order.broker_order_id,
            order.request,
            BrokerOrderStatus.CANCELLED,
            order.submitted_at,
        )
        return self.order


def test_qmt_adapter_state_prevents_duplicate_after_agent_restart() -> None:
    gateway = FakeQmtGateway()
    state = QmtAdapterState()
    request = _request()
    first = QmtBrokerAdapter(gateway, state).submit_order(request)
    second = QmtBrokerAdapter(gateway, state).submit_order(request)
    assert first == second
    assert gateway.submissions == 1
    assert QmtBrokerAdapter(gateway, state).cancel_order(first.broker_order_id).status is (
        BrokerOrderStatus.CANCELLED
    )


def test_reconciliation_treats_broker_snapshot_as_truth() -> None:
    local = _account(cash="100", quantity=100)
    broker = _account(cash="99", quantity=200)
    report = reconcile_accounts(local, broker)
    assert not report.matched
    assert {difference.field for difference in report.differences} == {"cash", "position"}
    assert report.broker_snapshot == broker
    assert reconcile_accounts(broker, broker).matched


def test_intraday_monitor_activates_kill_switch_and_reset_is_manual() -> None:
    kill = KillSwitch()
    monitor = IntradayRiskMonitor(
        kill,
        heartbeat_timeout=timedelta(seconds=10),
        maximum_drawdown=Decimal("0.05"),
    )
    monitor.evaluate(
        now=NOW,
        last_heartbeat=NOW - timedelta(seconds=11),
        opening_equity=Decimal("100"),
        current_equity=Decimal("90"),
        unknown_order_count=1,
    )
    assert kill.state.active and "UNKNOWN_ORDER_STATE" in (kill.state.reason or "")
    with pytest.raises(RuntimeError, match="active"):
        kill.require_inactive()
    with pytest.raises(ValueError, match="manual"):
        kill.reset(manual_confirmation=False)
    kill.reset(manual_confirmation=True)
    kill.require_inactive()
