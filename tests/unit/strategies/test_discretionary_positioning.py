from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal

import pytest
from tests.fixtures.discretionary import (
    DECISION_TIME,
    NOW,
    RELEASE_ID,
    SYMBOL,
    make_case,
    make_evidence,
)

from aquant.config import DiscretionaryStrategySettings
from aquant.domain import DataReleaseId, Symbol
from aquant.strategies.discretionary import (
    AllocationAction,
    AllocationTier,
    BayesianBeliefEngine,
    BeliefUpdate,
    DiscretionaryFundamentalStrategy,
    EstimateVintage,
    EvidenceFamily,
    EvidenceMaturity,
    MarketExpectationSnapshot,
    PayoffProfile,
    PositionPolicy,
    PositionSizingEngine,
    PriorPositionState,
    ResearchStage,
    ValidationDomain,
    approve_allocation,
    build_approved_target_portfolio,
)


def _expectation(
    *,
    implied_probability: Decimal = Decimal("0.20"),
    available_at=NOW + timedelta(hours=5),  # type: ignore[no-untyped-def]
) -> MarketExpectationSnapshot:
    return MarketExpectationSnapshot(
        "case-300394-ai-optics",
        "fundamental",
        implied_probability,
        Decimal("100"),
        "reverse DCF and consensus imply a low durable-growth probability",
        "research://market-expectation/v1",
        available_at - timedelta(minutes=1),
        available_at,
    )


def _payoff() -> PayoffProfile:
    return PayoffProfile(Decimal("0.60"), Decimal("0.30"), Decimal("0.50"))


def _observation_belief() -> BeliefUpdate:
    case = make_case()
    observations = (
        make_evidence("customer", cluster="customer"),
        make_evidence(
            "competitor",
            family=EvidenceFamily.COMPETITOR,
            cluster="competitor",
            variable_id="share",
            domains=(ValidationDomain.MARKET_SHARE,),
        ),
    )
    return BayesianBeliefEngine().update(
        case, observations, asof_time=DECISION_TIME, data_release_id=RELEASE_ID
    )


def test_position_policy_and_payoff_enforce_risk_invariants() -> None:
    with pytest.raises(ValueError, match="ordered"):
        PositionPolicy(observation_cap=Decimal("0.03"), evidence_cap=Decimal("0.02"))
    with pytest.raises(ValueError, match="human approval"):
        PositionPolicy(require_human_approval=False)
    with pytest.raises(ValueError, match="must be positive"):
        PayoffProfile(Decimal("0"), Decimal("0.2"), Decimal("0.5"))
    with pytest.raises(ValueError, match="between 0 and 1"):
        PayoffProfile(Decimal("0.5"), Decimal("0.2"), Decimal("1.1"))


def test_market_expectation_has_auditable_availability() -> None:
    with pytest.raises(ValueError, match="cannot precede"):
        replace(_expectation(), available_at=NOW)
    with pytest.raises(ValueError, match="positive"):
        replace(_expectation(), current_price=Decimal("0"))


def test_price_only_case_cannot_receive_a_position() -> None:
    case = make_case()
    price = make_evidence(
        "price-only",
        family=EvidenceFamily.PRICE,
        cluster="price",
        domains=(ValidationDomain.OTHER,),
    )
    belief = BayesianBeliefEngine().update(
        case, (price,), asof_time=DECISION_TIME, data_release_id=RELEASE_ID
    )

    proposal = PositionSizingEngine().propose(case, belief, _expectation(), _payoff())

    assert proposal.action is AllocationAction.WATCH
    assert proposal.tier is AllocationTier.NONE
    assert proposal.target_weight == 0
    assert "EVIDENCE_NOT_POSITION_ELIGIBLE" in proposal.reasons


def test_two_independent_non_price_clusters_allow_small_observation_position() -> None:
    case = make_case()
    belief = _observation_belief()
    expectation = _expectation()
    payoff = _payoff()
    policy = PositionPolicy()

    proposal = PositionSizingEngine(policy).propose(case, belief, expectation, payoff)

    assert proposal.action is AllocationAction.OPEN
    assert proposal.tier is AllocationTier.OBSERVATION
    assert Decimal("0") < proposal.target_weight <= Decimal("0.005")
    assert proposal.requires_manual_approval
    assert proposal.posterior_probability > proposal.market_probability
    assert proposal.belief_snapshot_id == belief.snapshot.snapshot_id
    assert proposal.market_expectation_id == expectation.expectation_id
    assert proposal.payoff_profile == payoff
    assert proposal.policy_hash == policy.policy_hash


def test_decision_identity_binds_market_payoff_and_policy_lineage() -> None:
    case = make_case()
    belief = _observation_belief()
    expectation = _expectation()
    first = PositionSizingEngine().propose(case, belief, expectation, _payoff())
    changed_expectation = replace(
        expectation, rationale="same probability, independently revised model"
    )
    second = PositionSizingEngine().propose(case, belief, changed_expectation, _payoff())
    changed_payoff = PositionSizingEngine().propose(
        case,
        belief,
        expectation,
        PayoffProfile(Decimal("0.61"), Decimal("0.30"), Decimal("0.50")),
    )

    assert first.proposal_id != second.proposal_id
    assert first.proposal_id != changed_payoff.proposal_id
    with pytest.raises(ValueError, match="belief_snapshot_id"):
        replace(first, belief_snapshot_id="invalid")


def test_market_pricing_or_bad_asymmetry_blocks_position() -> None:
    case = make_case()
    belief = _observation_belief()

    priced = PositionSizingEngine().propose(
        case, belief, _expectation(implied_probability=Decimal("0.99")), _payoff()
    )
    assert priced.target_weight == 0
    assert "INSUFFICIENT_MARKET_EXPECTATION_GAP" in priced.reasons

    bad_payoff = PayoffProfile(Decimal("0.01"), Decimal("0.90"), Decimal("0.5"))
    negative = PositionSizingEngine().propose(case, belief, _expectation(), bad_payoff)
    assert negative.target_weight == 0
    assert "NON_POSITIVE_EXPECTED_RETURN" in negative.reasons


def test_retroactive_or_future_information_cannot_size_a_position() -> None:
    case = make_case()
    real_time = _observation_belief()
    retrospective = BeliefUpdate(
        replace(real_time.snapshot, vintage=EstimateVintage.RETROSPECTIVE),
        real_time.assessment,
    )

    with pytest.raises(ValueError, match="retrospective"):
        PositionSizingEngine().propose(case, retrospective, _expectation(), _payoff())
    with pytest.raises(ValueError, match="future market expectations"):
        PositionSizingEngine().propose(
            case,
            real_time,
            _expectation(available_at=DECISION_TIME + timedelta(seconds=1)),
            _payoff(),
        )


def test_add_requires_a_new_and_material_evidence_upgrade() -> None:
    case = make_case()
    first = _observation_belief()
    prior = PriorPositionState(
        Decimal("0.001"),
        first.assessment.stage,
        first.assessment.evidence_strength,
        first.assessment.evidence_hash,
        first.assessment.validation_domains,
    )

    blocked = PositionSizingEngine().propose(case, first, _expectation(), _payoff(), prior=prior)
    assert blocked.target_weight == prior.current_weight
    assert blocked.action is AllocationAction.HOLD
    assert "ADD_BLOCKED_WITHOUT_EVIDENCE_UPGRADE" in blocked.reasons

    operating = make_evidence(
        "operating",
        family=EvidenceFamily.OFFICIAL_DISCLOSURE,
        cluster="operating-result",
        variable_id="profit",
        maturity=EvidenceMaturity.OPERATING_CONFIRMED,
        domains=(ValidationDomain.PROFITABILITY,),
    )
    upgraded = BayesianBeliefEngine().update(
        case,
        (
            make_evidence("customer", cluster="customer"),
            make_evidence(
                "competitor",
                family=EvidenceFamily.COMPETITOR,
                cluster="competitor",
                variable_id="share",
                domains=(ValidationDomain.MARKET_SHARE,),
            ),
            operating,
        ),
        asof_time=DECISION_TIME,
        data_release_id=RELEASE_ID,
    )
    added = PositionSizingEngine().propose(case, upgraded, _expectation(), _payoff(), prior=prior)
    assert added.action is AllocationAction.ADD
    assert added.target_weight > prior.current_weight
    assert added.tier is AllocationTier.EVIDENCE


def test_falsifier_exits_an_existing_position() -> None:
    case = make_case()
    falsifier = make_evidence(
        "falsifier",
        falsifies_fundamental=True,
        fundamental_lr=Decimal("0.05"),
        flow_lr=Decimal("5"),
    )
    belief = BayesianBeliefEngine().update(
        case, (falsifier,), asof_time=DECISION_TIME, data_release_id=RELEASE_ID
    )
    prior = PriorPositionState(
        Decimal("0.01"),
        ResearchStage.EVIDENCE,
        Decimal("0.5"),
        "a" * 64,
        (ValidationDomain.DEMAND,),
    )

    proposal = PositionSizingEngine().propose(case, belief, _expectation(), _payoff(), prior=prior)

    assert proposal.action is AllocationAction.EXIT
    assert proposal.target_weight == 0
    assert "FUNDAMENTAL_HYPOTHESIS_INVALIDATED" in proposal.reasons


def test_manual_approval_can_reduce_but_never_raise_recommended_weight() -> None:
    proposal = PositionSizingEngine().propose(
        make_case(), _observation_belief(), _expectation(), _payoff()
    )
    approval_time = DECISION_TIME + timedelta(hours=1)

    with pytest.raises(ValueError, match="manual confirmation"):
        approve_allocation(
            proposal,
            approved_by="researcher",
            rationale="reviewed raw evidence",
            approved_at=approval_time,
            manual_confirmation=False,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        approve_allocation(
            proposal,
            approved_by="researcher",
            rationale="too large",
            approved_at=approval_time,
            manual_confirmation=True,
            approved_weight=proposal.target_weight + Decimal("0.0001"),
        )

    approved = approve_allocation(
        proposal,
        approved_by="researcher",
        rationale="reviewed raw evidence and opposing case",
        approved_at=approval_time,
        manual_confirmation=True,
        approved_weight=Decimal("0.002"),
    )
    assert approved.approved_weight == Decimal("0.002")
    assert approved.manual_confirmation


def test_only_complete_fresh_approval_book_can_become_target_portfolio() -> None:
    proposal = PositionSizingEngine().propose(
        make_case(), _observation_belief(), _expectation(), _payoff()
    )
    approval_time = DECISION_TIME + timedelta(hours=1)
    approval = approve_allocation(
        proposal,
        approved_by="researcher",
        rationale="approved after adversarial review",
        approved_at=approval_time,
        manual_confirmation=True,
    )
    target_asof = approval_time + timedelta(minutes=1)

    with pytest.raises(ValueError, match="every existing"):
        build_approved_target_portfolio(
            (approval,),
            held_symbols=(Symbol.parse("000001.XSHE"),),
            strategy_id="discretionary-fundamental-v1",
            strategy_version="v1",
            trade_date=date(2026, 7, 24),
            asof_time=target_asof,
            data_release_id=RELEASE_ID,
        )
    with pytest.raises(ValueError, match="future approvals"):
        build_approved_target_portfolio(
            (approval,),
            held_symbols=(),
            strategy_id="discretionary-fundamental-v1",
            strategy_version="v1",
            trade_date=date(2026, 7, 24),
            asof_time=DECISION_TIME,
            data_release_id=RELEASE_ID,
        )
    with pytest.raises(ValueError, match="different data release"):
        build_approved_target_portfolio(
            (approval,),
            held_symbols=(),
            strategy_id="discretionary-fundamental-v1",
            strategy_version="v1",
            trade_date=date(2026, 7, 24),
            asof_time=target_asof,
            data_release_id=DataReleaseId("cn_equity_20260723_001"),
        )
    with pytest.raises(ValueError, match="duplicate symbols"):
        build_approved_target_portfolio(
            (approval, replace(approval, approval_id="b" * 64)),
            held_symbols=(),
            strategy_id="discretionary-fundamental-v1",
            strategy_version="v1",
            trade_date=date(2026, 7, 24),
            asof_time=target_asof,
            data_release_id=RELEASE_ID,
        )

    target = build_approved_target_portfolio(
        (approval,),
        held_symbols=(SYMBOL,),
        strategy_id="discretionary-fundamental-v1",
        strategy_version="v1",
        trade_date=date(2026, 7, 24),
        asof_time=target_asof,
        data_release_id=RELEASE_ID,
    )
    assert target.positions[0].symbol == SYMBOL
    assert target.positions[0].target_weight == approval.approved_weight
    assert target.signal_version.startswith("v1:")


def test_strategy_facade_stops_at_pending_allocation_proposal() -> None:
    case = make_case()
    observations = (
        make_evidence("customer", cluster="customer"),
        make_evidence(
            "competitor",
            family=EvidenceFamily.COMPETITOR,
            cluster="competitor",
            variable_id="share",
            domains=(ValidationDomain.MARKET_SHARE,),
        ),
    )

    decision = DiscretionaryFundamentalStrategy().review(
        case,
        observations,
        _expectation(),
        _payoff(),
        asof_time=DECISION_TIME,
        data_release_id=RELEASE_ID,
    )

    assert decision.proposal.requires_manual_approval
    assert decision.proposal.evidence_hash == decision.belief.snapshot.evidence_hash


def test_strategy_facade_uses_governed_runtime_configuration() -> None:
    with pytest.raises(ValueError, match="disabled by configuration"):
        DiscretionaryFundamentalStrategy.from_settings(DiscretionaryStrategySettings())

    settings = DiscretionaryStrategySettings(
        enabled=True,
        minimum_independent_non_price_clusters=3,
    )
    strategy = DiscretionaryFundamentalStrategy.from_settings(settings)
    case = make_case()
    decision = strategy.review(
        case,
        (
            make_evidence("customer", cluster="customer"),
            make_evidence(
                "competitor",
                family=EvidenceFamily.COMPETITOR,
                cluster="competitor",
                variable_id="share",
                domains=(ValidationDomain.MARKET_SHARE,),
            ),
        ),
        _expectation(),
        _payoff(),
        asof_time=DECISION_TIME,
        data_release_id=RELEASE_ID,
    )

    assert decision.belief.assessment.stage is ResearchStage.WATCHLIST
    assert decision.proposal.target_weight == 0
