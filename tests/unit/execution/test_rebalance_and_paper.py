from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.domain.portfolio import TargetPortfolio, TargetPosition
from aquant.execution import BrokerAdapter, BrokerOrderStatus, PaperBroker, PaperBrokerState
from aquant.portfolio import RebalancePlanner

SH = Symbol.parse("600000.XSHG")
SZ = Symbol.parse("000001.XSHE")


def _target() -> TargetPortfolio:
    return TargetPortfolio(
        "multi-factor",
        date(2026, 7, 17),
        datetime(2026, 7, 16, 7, tzinfo=UTC),
        "cn_equity_20260716_001",
        "signal-v1",
        (TargetPosition(SH, Decimal("0.5")), TargetPosition(SZ, Decimal("0.3"))),
    )


def test_rebalance_planner_sells_first_rounds_lots_and_is_deterministic() -> None:
    planner = RebalancePlanner()
    kwargs = {
        "account_id": "paper-001",
        "equity": Decimal("100000"),
        "current_quantities": {SH: 6000, SZ: 0},
        "prices": {SH: Decimal("10"), SZ: Decimal("20")},
    }
    first = planner.plan(_target(), **kwargs)  # type: ignore[arg-type]
    second = planner.plan(_target(), **kwargs)  # type: ignore[arg-type]
    assert [order.side for order in first.orders] == [Side.SELL, Side.BUY]
    assert [order.quantity for order in first.orders] == [1000, 1500]
    assert [order.idempotency_key for order in first.orders] == [
        order.idempotency_key for order in second.orders
    ]
    assert first.data_release_id == "cn_equity_20260716_001"


def test_paper_broker_executes_plan_and_restart_does_not_duplicate() -> None:
    plan = RebalancePlanner().plan(
        _target(),
        account_id="paper-001",
        equity=Decimal("100000"),
        current_quantities={},
        prices={SH: Decimal("10"), SZ: Decimal("20")},
    )
    state = PaperBrokerState(Decimal("100000"))
    broker = PaperBroker(state)
    assert isinstance(broker, BrokerAdapter)
    broker.set_price(SH, Decimal("10"))
    broker.set_price(SZ, Decimal("20"))
    orders = tuple(broker.submit_order(request) for request in plan.orders)
    assert all(order.status is BrokerOrderStatus.FILLED for order in orders)
    assert broker.query_account().positions == {SH: 5000, SZ: 1500}
    assert broker.query_account().cash == Decimal("20000")

    restarted = PaperBroker(state)
    repeated = tuple(restarted.submit_order(request) for request in plan.orders)
    assert repeated == orders
    assert len(restarted.query_fills()) == 2
    assert restarted.query_account().cash == Decimal("20000")


def test_paper_broker_rejects_insufficient_cash_and_oversell() -> None:
    plan = RebalancePlanner().plan(
        _target(),
        account_id="paper-001",
        equity=Decimal("100000"),
        current_quantities={},
        prices={SH: Decimal("10"), SZ: Decimal("20")},
    )
    state = PaperBrokerState(Decimal("1"))
    broker = PaperBroker(state)
    broker.set_price(SH, Decimal("10"))
    broker.set_price(SZ, Decimal("20"))
    rejected = broker.submit_order(plan.orders[0])
    assert rejected.status is BrokerOrderStatus.REJECTED

    sell_target = TargetPortfolio(
        "multi-factor",
        date(2026, 7, 17),
        datetime(2026, 7, 16, 7, tzinfo=UTC),
        "cn_equity_20260716_001",
        "signal-v2",
        (),
    )
    sell = (
        RebalancePlanner()
        .plan(
            sell_target,
            account_id="paper-001",
            equity=Decimal("100"),
            current_quantities={SH: 100},
            prices={SH: Decimal("10")},
        )
        .orders[0]
    )
    empty = PaperBroker(PaperBrokerState(Decimal("100")))
    empty.set_price(SH, Decimal("10"))
    assert empty.submit_order(sell).status is BrokerOrderStatus.REJECTED


def test_unpriced_paper_order_can_be_cancelled() -> None:
    request = (
        RebalancePlanner()
        .plan(
            _target(),
            account_id="paper-001",
            equity=Decimal("100000"),
            current_quantities={},
            prices={SH: Decimal("10"), SZ: Decimal("20")},
        )
        .orders[0]
    )
    broker = PaperBroker(PaperBrokerState(Decimal("100000")))
    submitted = broker.submit_order(request)
    assert submitted.status is BrokerOrderStatus.SUBMITTED
    assert broker.cancel_order(submitted.broker_order_id).status is BrokerOrderStatus.CANCELLED
    with pytest.raises(ValueError, match="submitted"):
        broker.cancel_order(submitted.broker_order_id)


def test_rebalance_validates_missing_price_and_inputs() -> None:
    with pytest.raises(KeyError, match="missing"):
        RebalancePlanner().plan(
            _target(),
            account_id="paper",
            equity=Decimal("1"),
            current_quantities={},
            prices={},
        )
    with pytest.raises(ValueError):
        PaperBroker(PaperBrokerState(Decimal("-1")))
