import hashlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from aquant.domain import DataReleaseId, Symbol
from aquant.strategies.discretionary import (
    AcquisitionBasis,
    CausalNode,
    CausalStage,
    CompetingHypothesis,
    EvidenceFamily,
    EvidenceMaturity,
    EvidenceObservation,
    EvidenceQuality,
    HypothesisKind,
    HypothesisLikelihood,
    KeyVariable,
    LeadingIndicator,
    RadarSignal,
    ResearchCase,
    ValidationDomain,
)

NOW = datetime(2026, 7, 22, 7, tzinfo=UTC)
DECISION_TIME = NOW + timedelta(days=1)
RELEASE_ID = DataReleaseId("cn_equity_20260722_001")
SYMBOL = Symbol.parse("300394.XSHE")


def _indicator(variable_id: str, suffix: str) -> LeadingIndicator:
    return LeadingIndicator(
        f"{variable_id}-{suffix}",
        f"{variable_id} indicator {suffix}",
        f"fixed-panel definition for {variable_id} {suffix}",
        "v1",
        (f"{variable_id}-{suffix}-primary", f"{variable_id}-{suffix}-cross-check"),
        60,
    )


def make_case() -> ResearchCase:
    variables = tuple(
        KeyVariable(
            variable_id,
            name,
            thesis,
            (_indicator(variable_id, "a"), _indicator(variable_id, "b")),
            (f"invalidate {variable_id}",),
        )
        for variable_id, name, thesis in (
            ("demand", "customer demand", "customer deployment accelerates"),
            ("share", "market share", "qualification converts into share"),
            ("profit", "profit and cash", "mix improvement reaches cash flow"),
        )
    )
    stage_indicators = {
        CausalStage.INDUSTRY_CHANGE: "demand-a",
        CausalStage.CUSTOMER_BEHAVIOR: "demand-b",
        CausalStage.COMPANY_ORDERS: "share-a",
        CausalStage.REVENUE: "share-b",
        CausalStage.MARGIN: "profit-a",
        CausalStage.CASH_FLOW: "profit-b",
        CausalStage.SHAREHOLDER_RETURN: "profit-b",
    }
    chain = tuple(
        CausalNode(
            stage,
            f"claim for {stage.value}",
            (f"known fact for {stage.value}",),
            (f"unknown for {stage.value}",),
            (stage_indicators[stage],),
            NOW + timedelta(days=30),
            f"falsifier for {stage.value}",
        )
        for stage in CausalStage
    )
    radar = RadarSignal(
        "radar-300394-20260722",
        SYMBOL,
        "RESIDUAL_RELATIVE_STRENGTH",
        "persistent unexplained relative strength",
        "radar-v1",
        NOW - timedelta(minutes=30),
        NOW - timedelta(minutes=20),
        RELEASE_ID,
    )
    hypotheses = (
        CompetingHypothesis(
            "fundamental",
            HypothesisKind.FUNDAMENTAL,
            "customer demand and share are improving",
            Decimal("0.40"),
            ("orders fail to convert",),
        ),
        CompetingHypothesis(
            "flow",
            HypothesisKind.FLOW_TECHNICAL,
            "theme rotation and positioning explain the move",
            Decimal("0.35"),
            ("strength persists after flow reverses",),
        ),
        CompetingHypothesis(
            "noise",
            HypothesisKind.NOISE,
            "the move is transient noise",
            Decimal("0.25"),
            ("independent operating evidence appears",),
        ),
    )
    return ResearchCase(
        "case-300394-ai-optics",
        SYMBOL,
        "AI optical upgrade",
        "How much durable growth is already priced in?",
        NOW,
        radar,
        hypotheses,
        chain,
        variables,
    )


def make_evidence(
    evidence_id: str,
    *,
    family: EvidenceFamily = EvidenceFamily.CUSTOMER,
    cluster: str = "customer-capex",
    variable_id: str = "demand",
    available_at: datetime = NOW + timedelta(hours=4),
    maturity: EvidenceMaturity = EvidenceMaturity.LEADING,
    domains: tuple[ValidationDomain, ...] = (ValidationDomain.DEMAND,),
    fundamental_lr: Decimal = Decimal("4"),
    flow_lr: Decimal = Decimal("0.7"),
    noise_lr: Decimal = Decimal("0.4"),
    reliability: Decimal = Decimal("0.9"),
    falsifies_fundamental: bool = False,
    source_id: str | None = None,
) -> EvidenceObservation:
    return EvidenceObservation(
        evidence_id,
        "case-300394-ai-optics",
        variable_id,
        source_id or f"source-{evidence_id}",
        f"https://example.test/{evidence_id}",
        family,
        cluster,
        AcquisitionBasis.PUBLIC,
        "v1",
        f"summary for {evidence_id}",
        available_at - timedelta(minutes=3),
        available_at - timedelta(minutes=2),
        available_at - timedelta(minutes=1),
        available_at,
        hashlib.sha256(evidence_id.encode()).hexdigest(),
        maturity,
        domains,
        EvidenceQuality(
            Decimal("0.9"),
            reliability,
            Decimal("0.9"),
            Decimal("0.9"),
            Decimal("0.9"),
        ),
        (
            HypothesisLikelihood("fundamental", fundamental_lr),
            HypothesisLikelihood("flow", flow_lr),
            HypothesisLikelihood("noise", noise_lr),
        ),
        falsifies_fundamental,
    )
