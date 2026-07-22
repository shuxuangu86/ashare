from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from tests.fixtures.discretionary import (
    DECISION_TIME,
    RELEASE_ID,
    make_case,
    make_evidence,
)

from aquant.strategies.discretionary import (
    BayesianBeliefEngine,
    CausalNode,
    CausalStage,
    CompetingHypothesis,
    EstimateVintage,
    EvidenceAssessment,
    EvidenceFamily,
    EvidenceMaturity,
    EvidenceQuality,
    HypothesisKind,
    HypothesisLikelihood,
    HypothesisProbability,
    LeadingIndicator,
    ResearchStage,
    ValidationDomain,
)


def test_research_case_encodes_competing_hypotheses_and_full_causal_chain() -> None:
    case = make_case()

    assert case.fundamental_hypothesis_id == "fundamental"
    assert tuple(node.stage for node in case.causal_chain) == tuple(CausalStage)
    assert len(case.key_variables) == 3
    assert all(len(variable.leading_indicators) >= 2 for variable in case.key_variables)


def test_research_case_rejects_incomplete_or_incoherent_structure() -> None:
    case = make_case()

    with pytest.raises(ValueError, match="exactly three"):
        replace(case, key_variables=case.key_variables[:2])
    with pytest.raises(ValueError, match="complete ordered"):
        replace(case, causal_chain=tuple(reversed(case.causal_chain)))
    with pytest.raises(ValueError, match="exactly three competing"):
        replace(case, hypotheses=case.hypotheses[:2])
    wrong_kinds = (
        case.hypotheses[0],
        replace(case.hypotheses[1], kind=HypothesisKind.FUNDAMENTAL),
        case.hypotheses[2],
    )
    with pytest.raises(ValueError, match="fundamental, flow/technical, and noise"):
        replace(case, hypotheses=wrong_kinds)
    bad_priors = (
        replace(case.hypotheses[0], prior_probability=Decimal("0.5")),
        case.hypotheses[1],
        case.hypotheses[2],
    )
    with pytest.raises(ValueError, match="sum to 1"):
        replace(case, hypotheses=bad_priors)
    bad_node = replace(case.causal_chain[0], indicator_ids=("undefined",))
    with pytest.raises(ValueError, match="undefined leading indicator"):
        replace(case, causal_chain=(bad_node, *case.causal_chain[1:]))
    with pytest.raises(ValueError, match="cannot predate"):
        replace(case, created_at=case.radar_signal.available_at - timedelta(seconds=1))


def test_hypothesis_indicator_and_causal_node_invariants() -> None:
    with pytest.raises(ValueError, match="falsification"):
        CompetingHypothesis("h", HypothesisKind.NOISE, "noise", Decimal("0.2"), ())
    with pytest.raises(ValueError, match="greater than 0"):
        CompetingHypothesis("h", HypothesisKind.NOISE, "noise", Decimal("0"), ("falsifier",))
    with pytest.raises(ValueError, match="two independent"):
        LeadingIndicator("i", "name", "definition", "v1", ("one",), 1)
    with pytest.raises(ValueError, match="cannot be negative"):
        LeadingIndicator("i", "name", "definition", "v1", ("one", "two"), -1)
    with pytest.raises(ValueError, match="explicit unknown"):
        CausalNode(
            CausalStage.REVENUE,
            "claim",
            (),
            (),
            ("demand-a",),
            DECISION_TIME,
            "falsifier",
        )


def test_evidence_requires_strict_vintage_quality_and_complete_likelihoods() -> None:
    evidence = make_evidence("valid")

    with pytest.raises(ValueError, match="observed <= published"):
        replace(evidence, published_at=evidence.observed_at - timedelta(seconds=1))
    with pytest.raises(ValueError, match="validation domain"):
        replace(evidence, validation_domains=())
    with pytest.raises(ValueError, match="non-empty and unique"):
        replace(evidence, likelihoods=(evidence.likelihoods[0], evidence.likelihoods[0]))
    with pytest.raises(ValueError, match=r"between 0\.05 and 20"):
        HypothesisLikelihood("fundamental", Decimal("30"))
    with pytest.raises(ValueError, match="between 0 and 1"):
        EvidenceQuality(
            Decimal("1.1"),
            Decimal("0.5"),
            Decimal("0.5"),
            Decimal("0.5"),
            Decimal("0.5"),
        )


def test_no_evidence_keeps_priors_and_radar_stage() -> None:
    case = make_case()

    update = BayesianBeliefEngine().update(
        case, (), asof_time=DECISION_TIME, data_release_id=RELEASE_ID
    )

    assert update.assessment.stage is ResearchStage.RADAR
    assert update.assessment.evidence_strength == 0
    assert update.snapshot.probability_for("fundamental") == Decimal("0.40")
    with pytest.raises(KeyError, match="absent"):
        update.snapshot.probability_for("missing")


def test_evidence_breadth_gate_is_configurable_but_never_below_two_clusters() -> None:
    case = make_case()
    observations = (
        make_evidence("customer", cluster="customer"),
        make_evidence(
            "competitor",
            family=EvidenceFamily.COMPETITOR,
            cluster="competitor",
            variable_id="share",
        ),
    )

    update = BayesianBeliefEngine(minimum_non_price_clusters=3).update(
        case, observations, asof_time=DECISION_TIME, data_release_id=RELEASE_ID
    )

    assert update.assessment.stage is ResearchStage.WATCHLIST
    with pytest.raises(ValueError, match="at least two"):
        BayesianBeliefEngine(minimum_non_price_clusters=1)


def test_price_is_a_research_trigger_but_never_position_grade_evidence() -> None:
    case = make_case()
    price = make_evidence(
        "price",
        family=EvidenceFamily.PRICE,
        cluster="price-flow",
        domains=(ValidationDomain.OTHER,),
    )

    update = BayesianBeliefEngine().update(
        case, (price,), asof_time=DECISION_TIME, data_release_id=RELEASE_ID
    )

    assert update.assessment.stage is ResearchStage.HYPOTHESIS
    assert update.assessment.non_price_clusters == ()
    assert update.assessment.evidence_strength == 0
    assert update.snapshot.probability_for("fundamental") > Decimal("0.40")


def test_future_evidence_is_invisible_to_real_time_snapshot() -> None:
    case = make_case()
    visible = make_evidence("visible")
    future = make_evidence("future", available_at=DECISION_TIME + timedelta(seconds=1))

    update = BayesianBeliefEngine().update(
        case,
        (visible, future),
        asof_time=DECISION_TIME,
        data_release_id=RELEASE_ID,
        vintage=EstimateVintage.REAL_TIME,
    )

    assert update.snapshot.evidence_ids == ("visible",)
    assert update.assessment.evidence_ids == ("visible",)


def test_same_dependency_cluster_is_not_double_counted() -> None:
    case = make_case()
    first = make_evidence("first", cluster="same-root")
    duplicate = make_evidence("duplicate", cluster="same-root")
    engine = BayesianBeliefEngine()

    one = engine.update(case, (first,), asof_time=DECISION_TIME, data_release_id=RELEASE_ID)
    two = engine.update(
        case, (first, duplicate), asof_time=DECISION_TIME, data_release_id=RELEASE_ID
    )

    assert two.snapshot.probability_for("fundamental") == one.snapshot.probability_for(
        "fundamental"
    )
    assert two.assessment.independent_clusters == ("same-root",)


def test_independent_operating_and_financial_evidence_promotes_in_stages() -> None:
    case = make_case()
    demand = make_evidence("demand", cluster="customer", domains=(ValidationDomain.DEMAND,))
    share = make_evidence(
        "share",
        family=EvidenceFamily.COMPETITOR,
        cluster="competitor",
        variable_id="share",
        domains=(ValidationDomain.MARKET_SHARE,),
    )
    engine = BayesianBeliefEngine()

    observation = engine.update(
        case, (demand, share), asof_time=DECISION_TIME, data_release_id=RELEASE_ID
    )
    assert observation.assessment.stage is ResearchStage.OBSERVATION
    assert observation.assessment.evidence_strength > 0

    operating = replace(share, maturity=EvidenceMaturity.OPERATING_CONFIRMED)
    evidence = engine.update(
        case, (demand, operating), asof_time=DECISION_TIME, data_release_id=RELEASE_ID
    )
    assert evidence.assessment.stage is ResearchStage.EVIDENCE

    profit = make_evidence(
        "profit",
        family=EvidenceFamily.OFFICIAL_DISCLOSURE,
        cluster="profit-report",
        variable_id="profit",
        domains=(ValidationDomain.PROFITABILITY,),
        maturity=EvidenceMaturity.FINANCIAL_CONFIRMED,
    )
    cash = make_evidence(
        "cash",
        family=EvidenceFamily.FINANCIAL_RESULT,
        cluster="cash-flow",
        variable_id="profit",
        domains=(ValidationDomain.CASH_FLOW,),
        maturity=EvidenceMaturity.FINANCIAL_CONFIRMED,
    )
    core = engine.update(
        case,
        (demand, operating, profit, cash),
        asof_time=DECISION_TIME,
        data_release_id=RELEASE_ID,
    )
    assert core.assessment.stage is ResearchStage.CORE
    assert set(core.assessment.validation_domains) >= {
        ValidationDomain.DEMAND,
        ValidationDomain.MARKET_SHARE,
        ValidationDomain.PROFITABILITY,
        ValidationDomain.CASH_FLOW,
    }


def test_explicit_falsifier_invalidates_case() -> None:
    case = make_case()
    falsifier = make_evidence(
        "falsifier",
        fundamental_lr=Decimal("0.05"),
        flow_lr=Decimal("5"),
        noise_lr=Decimal("2"),
        falsifies_fundamental=True,
    )

    update = BayesianBeliefEngine().update(
        case, (falsifier,), asof_time=DECISION_TIME, data_release_id=RELEASE_ID
    )

    assert update.assessment.stage is ResearchStage.INVALIDATED
    assert update.assessment.falsifier_seen


def test_belief_engine_rejects_bad_lineage() -> None:
    case = make_case()
    evidence = make_evidence("lineage")

    with pytest.raises(ValueError, match="duplicate evidence"):
        BayesianBeliefEngine().update(
            case,
            (evidence, evidence),
            asof_time=DECISION_TIME,
            data_release_id=RELEASE_ID,
        )
    with pytest.raises(ValueError, match="different research case"):
        BayesianBeliefEngine().update(
            case,
            (replace(evidence, case_id="other"),),
            asof_time=DECISION_TIME,
            data_release_id=RELEASE_ID,
        )
    with pytest.raises(ValueError, match="undefined key variable"):
        BayesianBeliefEngine().update(
            case,
            (replace(evidence, variable_id="undefined"),),
            asof_time=DECISION_TIME,
            data_release_id=RELEASE_ID,
        )
    with pytest.raises(ValueError, match="every competing hypothesis"):
        BayesianBeliefEngine().update(
            case,
            (replace(evidence, likelihoods=evidence.likelihoods[:2]),),
            asof_time=DECISION_TIME,
            data_release_id=RELEASE_ID,
        )


def test_probability_and_assessment_value_objects_reject_incoherent_values() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        HypothesisProbability("h", Decimal("1.1"))
    with pytest.raises(ValueError, match="subset"):
        EvidenceAssessment(
            "case",
            DECISION_TIME,
            ResearchStage.WATCHLIST,
            Decimal("0.2"),
            ("a",),
            ("b",),
            (ValidationDomain.OTHER,),
            (),
            "a" * 64,
            False,
        )
