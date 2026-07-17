from datetime import UTC, date, datetime
from decimal import Decimal

from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import LimitStatus, SecurityStatus
from aquant.execution import (
    BrokerAccountSnapshot,
    ExecutionAgent,
    ExecutionOrderStore,
    KillSwitch,
    PaperBroker,
    PaperBrokerState,
    reconcile_accounts,
)
from aquant.portfolio import (
    MultiFactorScorer,
    PreTradeContext,
    PreTradeRiskEngine,
    RebalancePlanner,
    construct_top_n_equal_weight,
)

SYMBOLS = tuple(Symbol.parse(value) for value in ("600000.XSHG", "000001.XSHE"))
NOW = datetime(2026, 7, 16, 7, tzinfo=UTC)


def test_factor_to_target_to_risk_to_paper_to_reconciliation() -> None:
    scores = MultiFactorScorer().score(
        {
            "momentum": {SYMBOLS[0]: 1.0, SYMBOLS[1]: 2.0},
            "value": {SYMBOLS[0]: 1.5, SYMBOLS[1]: 1.0},
        },
        {"momentum": 1.0, "value": 0.2},
    )
    target = construct_top_n_equal_weight(
        scores,
        top_n=1,
        maximum_weight=Decimal("0.5"),
        minimum_cash_weight=Decimal("0.5"),
        strategy_id="full-chain",
        trade_date=date(2026, 7, 17),
        asof_time=NOW,
        data_release_id="cn_equity_20260716_001",
        signal_version="v1",
    )
    prices = {SYMBOLS[0]: Decimal("10"), SYMBOLS[1]: Decimal("20")}
    plan = RebalancePlanner().plan(
        target,
        account_id="paper-001",
        equity=Decimal("100000"),
        current_quantities={},
        prices=prices,
    )
    state = PaperBrokerState(Decimal("100000"))
    broker = PaperBroker(state)
    for symbol, price in prices.items():
        broker.set_price(symbol, price)
    store = ExecutionOrderStore()
    for request in plan.orders:
        store.approve(request)

    def context(request: object) -> PreTradeContext:
        symbol = request.symbol  # type: ignore[attr-defined]
        return PreTradeContext(
            NOW,
            NOW,
            prices[symbol],
            Decimal("1000000"),
            SecurityStatus(symbol, date(2026, 7, 17), False, False, LimitStatus.NONE),
            broker.query_account(),
            True,
        )

    result = ExecutionAgent(
        store=store,
        broker=broker,
        risk_engine=PreTradeRiskEngine(),
        context_provider=context,  # type: ignore[arg-type]
        kill_switch=KillSwitch(),
    ).run_once()
    assert result.submitted == 1
    broker_snapshot = broker.query_account()
    local_snapshot = BrokerAccountSnapshot(
        broker_snapshot.asof_time,
        broker_snapshot.cash,
        dict(broker_snapshot.positions),
    )
    assert reconcile_accounts(local_snapshot, broker_snapshot).matched
    assert sum(broker_snapshot.positions.values()) > 0
