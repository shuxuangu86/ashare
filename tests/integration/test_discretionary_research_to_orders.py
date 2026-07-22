from datetime import date, timedelta
from decimal import Decimal

from tests.fixtures.discretionary import (
    DECISION_TIME,
    NOW,
    RELEASE_ID,
    SYMBOL,
    make_case,
    make_evidence,
)

from aquant.domain import Side
from aquant.portfolio import RebalancePlanner
from aquant.strategies import DiscretionaryFundamentalStrategy
from aquant.strategies.discretionary import (
    EvidenceFamily,
    MarketExpectationSnapshot,
    PayoffProfile,
    ValidationDomain,
    approve_allocation,
    build_approved_target_portfolio,
)


def test_discretionary_evidence_to_human_approval_to_standard_order_intent() -> None:
    case = make_case()
    evidence = (
        make_evidence("customer", cluster="customer"),
        make_evidence(
            "competitor",
            family=EvidenceFamily.COMPETITOR,
            cluster="competitor",
            variable_id="share",
            domains=(ValidationDomain.MARKET_SHARE,),
        ),
    )
    expectation = MarketExpectationSnapshot(
        case.case_id,
        case.fundamental_hypothesis_id,
        Decimal("0.2"),
        Decimal("10"),
        "market prices only a short demand cycle",
        "research://expectation/integration",
        NOW + timedelta(hours=4),
        NOW + timedelta(hours=5),
    )
    decision = DiscretionaryFundamentalStrategy().review(
        case,
        evidence,
        expectation,
        PayoffProfile(Decimal("0.6"), Decimal("0.3"), Decimal("0.5")),
        asof_time=DECISION_TIME,
        data_release_id=RELEASE_ID,
    )
    approval_time = DECISION_TIME + timedelta(minutes=5)
    approval = approve_allocation(
        decision.proposal,
        approved_by="human-researcher",
        rationale="raw sources and opposing hypotheses reviewed",
        approved_at=approval_time,
        manual_confirmation=True,
    )
    target = build_approved_target_portfolio(
        (approval,),
        held_symbols=(),
        strategy_id="discretionary-fundamental-v1",
        strategy_version="v1",
        trade_date=date(2026, 7, 24),
        asof_time=approval_time + timedelta(minutes=1),
        data_release_id=RELEASE_ID,
    )

    plan = RebalancePlanner().plan(
        target,
        account_id="paper-subjective",
        equity=Decimal("1000000"),
        current_quantities={},
        prices={SYMBOL: Decimal("10")},
    )

    assert len(plan.orders) == 1
    assert plan.orders[0].side is Side.BUY
    assert plan.orders[0].quantity % 100 == 0
    assert plan.strategy_id == "discretionary-fundamental-v1"
