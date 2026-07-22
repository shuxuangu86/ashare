import hashlib
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from aquant.domain.data_release import DataReleaseId
from aquant.domain.identifiers import Symbol
from aquant.domain.time import require_aware


def _require_text(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _require_probability(value: Decimal, *, field_name: str, allow_zero: bool = True) -> Decimal:
    normalized = Decimal(value)
    lower_bound_ok = normalized >= 0 if allow_zero else normalized > 0
    if not lower_bound_ok or normalized > 1:
        qualifier = "between 0 and 1" if allow_zero else "greater than 0 and at most 1"
        raise ValueError(f"{field_name} must be {qualifier}")
    return normalized


def _require_sha256(value: str, *, field_name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    return normalized


class HypothesisKind(StrEnum):
    FUNDAMENTAL = "FUNDAMENTAL"
    FLOW_TECHNICAL = "FLOW_TECHNICAL"
    NOISE = "NOISE"


class CausalStage(StrEnum):
    INDUSTRY_CHANGE = "INDUSTRY_CHANGE"
    CUSTOMER_BEHAVIOR = "CUSTOMER_BEHAVIOR"
    COMPANY_ORDERS = "COMPANY_ORDERS"
    REVENUE = "REVENUE"
    MARGIN = "MARGIN"
    CASH_FLOW = "CASH_FLOW"
    SHAREHOLDER_RETURN = "SHAREHOLDER_RETURN"


CAUSAL_STAGE_ORDER = tuple(CausalStage)


class EvidenceFamily(StrEnum):
    PRICE = "PRICE"
    CUSTOMER = "CUSTOMER"
    SUPPLIER = "SUPPLIER"
    COMPANY_ACTION = "COMPANY_ACTION"
    CHANNEL = "CHANNEL"
    COMPETITOR = "COMPETITOR"
    OFFICIAL_DISCLOSURE = "OFFICIAL_DISCLOSURE"
    FINANCIAL_RESULT = "FINANCIAL_RESULT"


class EvidenceMaturity(StrEnum):
    LEADING = "LEADING"
    OPERATING_CONFIRMED = "OPERATING_CONFIRMED"
    FINANCIAL_CONFIRMED = "FINANCIAL_CONFIRMED"


class ValidationDomain(StrEnum):
    DEMAND = "DEMAND"
    MARKET_SHARE = "MARKET_SHARE"
    PROFITABILITY = "PROFITABILITY"
    CASH_FLOW = "CASH_FLOW"
    OTHER = "OTHER"


CORE_VALIDATION_DOMAINS = frozenset(
    {
        ValidationDomain.DEMAND,
        ValidationDomain.MARKET_SHARE,
        ValidationDomain.PROFITABILITY,
        ValidationDomain.CASH_FLOW,
    }
)


class AcquisitionBasis(StrEnum):
    PUBLIC = "PUBLIC"
    LICENSED = "LICENSED"
    COMPLIANT_RESEARCH = "COMPLIANT_RESEARCH"


class EstimateVintage(StrEnum):
    REAL_TIME = "REAL_TIME"
    RETROSPECTIVE = "RETROSPECTIVE"


class ResearchStage(StrEnum):
    RADAR = "RADAR"
    HYPOTHESIS = "HYPOTHESIS"
    WATCHLIST = "WATCHLIST"
    OBSERVATION = "OBSERVATION"
    EVIDENCE = "EVIDENCE"
    CORE = "CORE"
    INVALIDATED = "INVALIDATED"


class OutcomeLayer(StrEnum):
    MEASUREMENT = "MEASUREMENT"
    TRANSMISSION = "TRANSMISSION"
    INVESTMENT = "INVESTMENT"


@dataclass(frozen=True, slots=True)
class RadarSignal:
    signal_id: str
    symbol: Symbol
    signal_type: str
    description: str
    model_version: str
    observed_at: datetime
    available_at: datetime
    data_release_id: DataReleaseId

    def __post_init__(self) -> None:
        observed_at = require_aware(self.observed_at, field_name="observed_at")
        available_at = require_aware(self.available_at, field_name="available_at")
        if available_at < observed_at:
            raise ValueError("radar signal available_at cannot precede observed_at")
        for field_name in ("signal_id", "signal_type", "description", "model_version"):
            object.__setattr__(
                self, field_name, _require_text(getattr(self, field_name), field_name=field_name)
            )
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "available_at", available_at)


@dataclass(frozen=True, slots=True)
class CompetingHypothesis:
    hypothesis_id: str
    kind: HypothesisKind
    statement: str
    prior_probability: Decimal
    falsification_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.falsification_conditions or any(
            not condition.strip() for condition in self.falsification_conditions
        ):
            raise ValueError("hypothesis requires explicit falsification conditions")
        object.__setattr__(
            self, "hypothesis_id", _require_text(self.hypothesis_id, field_name="hypothesis_id")
        )
        object.__setattr__(self, "statement", _require_text(self.statement, field_name="statement"))
        object.__setattr__(
            self,
            "prior_probability",
            _require_probability(
                self.prior_probability, field_name="prior_probability", allow_zero=False
            ),
        )
        object.__setattr__(
            self,
            "falsification_conditions",
            tuple(condition.strip() for condition in self.falsification_conditions),
        )


@dataclass(frozen=True, slots=True)
class LeadingIndicator:
    indicator_id: str
    name: str
    definition: str
    method_version: str
    source_cluster_ids: tuple[str, ...]
    expected_lead_days: int

    def __post_init__(self) -> None:
        clusters = tuple(cluster.strip() for cluster in self.source_cluster_ids if cluster.strip())
        if len(clusters) < 2 or len(clusters) != len(set(clusters)):
            raise ValueError("leading indicator requires at least two independent source clusters")
        if self.expected_lead_days < 0:
            raise ValueError("expected_lead_days cannot be negative")
        for field_name in ("indicator_id", "name", "definition", "method_version"):
            object.__setattr__(
                self, field_name, _require_text(getattr(self, field_name), field_name=field_name)
            )
        object.__setattr__(self, "source_cluster_ids", clusters)


@dataclass(frozen=True, slots=True)
class KeyVariable:
    variable_id: str
    name: str
    thesis: str
    leading_indicators: tuple[LeadingIndicator, ...]
    falsification_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.leading_indicators) < 2:
            raise ValueError("key variable requires at least two leading indicators")
        indicator_ids = [indicator.indicator_id for indicator in self.leading_indicators]
        if len(indicator_ids) != len(set(indicator_ids)):
            raise ValueError("key variable cannot contain duplicate indicator ids")
        if not self.falsification_conditions or any(
            not condition.strip() for condition in self.falsification_conditions
        ):
            raise ValueError("key variable requires falsification conditions")
        for field_name in ("variable_id", "name", "thesis"):
            object.__setattr__(
                self, field_name, _require_text(getattr(self, field_name), field_name=field_name)
            )
        object.__setattr__(
            self,
            "falsification_conditions",
            tuple(condition.strip() for condition in self.falsification_conditions),
        )


@dataclass(frozen=True, slots=True)
class CausalNode:
    stage: CausalStage
    claim: str
    known_facts: tuple[str, ...]
    unknowns: tuple[str, ...]
    indicator_ids: tuple[str, ...]
    next_validation_at: datetime
    falsification_condition: str

    def __post_init__(self) -> None:
        if not self.unknowns or any(not value.strip() for value in self.unknowns):
            raise ValueError("causal node requires at least one explicit unknown")
        indicators = tuple(value.strip() for value in self.indicator_ids if value.strip())
        if not indicators:
            raise ValueError("causal node requires at least one leading indicator")
        if len(indicators) != len(set(indicators)):
            raise ValueError("causal node cannot contain duplicate indicator ids")
        if any(not value.strip() for value in self.known_facts):
            raise ValueError("known facts must not contain blank entries")
        object.__setattr__(self, "claim", _require_text(self.claim, field_name="claim"))
        object.__setattr__(
            self,
            "falsification_condition",
            _require_text(self.falsification_condition, field_name="falsification_condition"),
        )
        object.__setattr__(self, "known_facts", tuple(value.strip() for value in self.known_facts))
        object.__setattr__(self, "unknowns", tuple(value.strip() for value in self.unknowns))
        object.__setattr__(self, "indicator_ids", indicators)
        object.__setattr__(
            self,
            "next_validation_at",
            require_aware(self.next_validation_at, field_name="next_validation_at"),
        )


@dataclass(frozen=True, slots=True)
class ResearchCase:
    case_id: str
    symbol: Symbol
    title: str
    market_question: str
    created_at: datetime
    radar_signal: RadarSignal
    hypotheses: tuple[CompetingHypothesis, ...]
    causal_chain: tuple[CausalNode, ...]
    key_variables: tuple[KeyVariable, ...]

    def __post_init__(self) -> None:
        created_at = require_aware(self.created_at, field_name="created_at")
        if self.radar_signal.symbol != self.symbol:
            raise ValueError("research case and radar signal symbols must match")
        if self.radar_signal.available_at > created_at:
            raise ValueError("research case cannot predate its radar signal")
        if len(self.key_variables) != 3:
            raise ValueError("research case must identify exactly three key variables")
        variable_ids = [variable.variable_id for variable in self.key_variables]
        if len(variable_ids) != len(set(variable_ids)):
            raise ValueError("research case cannot contain duplicate key variable ids")

        if len(self.hypotheses) != 3:
            raise ValueError("research case requires exactly three competing hypotheses")
        hypothesis_ids = [hypothesis.hypothesis_id for hypothesis in self.hypotheses]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("research case cannot contain duplicate hypothesis ids")
        kinds = {hypothesis.kind for hypothesis in self.hypotheses}
        if kinds != set(HypothesisKind):
            raise ValueError(
                "research case requires fundamental, flow/technical, and noise hypotheses"
            )
        prior_total = sum(
            (hypothesis.prior_probability for hypothesis in self.hypotheses), Decimal("0")
        )
        if prior_total != Decimal("1"):
            raise ValueError("hypothesis prior probabilities must sum to 1")

        stages = tuple(node.stage for node in self.causal_chain)
        if stages != CAUSAL_STAGE_ORDER:
            raise ValueError("research case requires the complete ordered causal chain")
        defined_indicators = {
            indicator.indicator_id
            for variable in self.key_variables
            for indicator in variable.leading_indicators
        }
        referenced_indicators = {
            indicator_id for node in self.causal_chain for indicator_id in node.indicator_ids
        }
        if not referenced_indicators <= defined_indicators:
            raise ValueError("causal chain references an undefined leading indicator")

        for field_name in ("case_id", "title", "market_question"):
            object.__setattr__(
                self, field_name, _require_text(getattr(self, field_name), field_name=field_name)
            )
        object.__setattr__(self, "created_at", created_at)

    @property
    def fundamental_hypothesis_id(self) -> str:
        return next(
            hypothesis.hypothesis_id
            for hypothesis in self.hypotheses
            if hypothesis.kind is HypothesisKind.FUNDAMENTAL
        )


@dataclass(frozen=True, slots=True)
class EvidenceQuality:
    causal_proximity: Decimal
    reliability: Decimal
    independence: Decimal
    timeliness: Decimal
    expectation_gap: Decimal

    def __post_init__(self) -> None:
        for field_name in (
            "causal_proximity",
            "reliability",
            "independence",
            "timeliness",
            "expectation_gap",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_probability(getattr(self, field_name), field_name=field_name),
            )

    def combined(self, *, reliability: Decimal | None = None) -> Decimal:
        effective_reliability = (
            self.reliability
            if reliability is None
            else _require_probability(reliability, field_name="reliability")
        )
        return (
            self.causal_proximity
            * effective_reliability
            * self.independence
            * self.timeliness
            * self.expectation_gap
        )


@dataclass(frozen=True, slots=True)
class HypothesisLikelihood:
    hypothesis_id: str
    likelihood_ratio: Decimal

    def __post_init__(self) -> None:
        ratio = Decimal(self.likelihood_ratio)
        if not Decimal("0.05") <= ratio <= Decimal("20"):
            raise ValueError("likelihood_ratio must be between 0.05 and 20")
        object.__setattr__(
            self, "hypothesis_id", _require_text(self.hypothesis_id, field_name="hypothesis_id")
        )
        object.__setattr__(self, "likelihood_ratio", ratio)


@dataclass(frozen=True, slots=True)
class EvidenceObservation:
    evidence_id: str
    case_id: str
    variable_id: str
    source_id: str
    source_uri: str
    family: EvidenceFamily
    dependency_cluster_id: str
    acquisition_basis: AcquisitionBasis
    method_version: str
    summary: str
    observed_at: datetime
    published_at: datetime
    collected_at: datetime
    available_at: datetime
    raw_content_sha256: str
    maturity: EvidenceMaturity
    validation_domains: tuple[ValidationDomain, ...]
    quality: EvidenceQuality
    likelihoods: tuple[HypothesisLikelihood, ...]
    falsifies_fundamental: bool = False

    def __post_init__(self) -> None:
        observed_at = require_aware(self.observed_at, field_name="observed_at")
        published_at = require_aware(self.published_at, field_name="published_at")
        collected_at = require_aware(self.collected_at, field_name="collected_at")
        available_at = require_aware(self.available_at, field_name="available_at")
        if not observed_at <= published_at <= collected_at <= available_at:
            raise ValueError(
                "evidence timestamps must satisfy observed <= published <= collected <= available"
            )
        domains = tuple(dict.fromkeys(self.validation_domains))
        if not domains:
            raise ValueError("evidence requires at least one validation domain")
        hypothesis_ids = [likelihood.hypothesis_id for likelihood in self.likelihoods]
        if not hypothesis_ids or len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("evidence likelihood hypothesis ids must be non-empty and unique")
        for field_name in (
            "evidence_id",
            "case_id",
            "variable_id",
            "source_id",
            "source_uri",
            "dependency_cluster_id",
            "method_version",
            "summary",
        ):
            object.__setattr__(
                self, field_name, _require_text(getattr(self, field_name), field_name=field_name)
            )
        object.__setattr__(
            self,
            "raw_content_sha256",
            _require_sha256(self.raw_content_sha256, field_name="raw_content_sha256"),
        )
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "published_at", published_at)
        object.__setattr__(self, "collected_at", collected_at)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "validation_domains", domains)


@dataclass(frozen=True, slots=True)
class HypothesisProbability:
    hypothesis_id: str
    probability: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "hypothesis_id", _require_text(self.hypothesis_id, field_name="hypothesis_id")
        )
        object.__setattr__(
            self,
            "probability",
            _require_probability(self.probability, field_name="probability"),
        )


@dataclass(frozen=True, slots=True)
class BeliefSnapshot:
    case_id: str
    asof_time: datetime
    data_release_id: DataReleaseId
    vintage: EstimateVintage
    probabilities: tuple[HypothesisProbability, ...]
    evidence_ids: tuple[str, ...]
    evidence_hash: str

    def __post_init__(self) -> None:
        probabilities = [item.probability for item in self.probabilities]
        ids = [item.hypothesis_id for item in self.probabilities]
        if not probabilities or len(ids) != len(set(ids)):
            raise ValueError("belief snapshot hypothesis ids must be non-empty and unique")
        if sum(probabilities, Decimal("0")) != Decimal("1"):
            raise ValueError("belief snapshot probabilities must sum to 1")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("belief snapshot cannot contain duplicate evidence ids")
        object.__setattr__(self, "case_id", _require_text(self.case_id, field_name="case_id"))
        object.__setattr__(self, "asof_time", require_aware(self.asof_time, field_name="asof_time"))
        object.__setattr__(
            self,
            "evidence_hash",
            _require_sha256(self.evidence_hash, field_name="evidence_hash"),
        )

    def probability_for(self, hypothesis_id: str) -> Decimal:
        for item in self.probabilities:
            if item.hypothesis_id == hypothesis_id:
                return item.probability
        raise KeyError(f"hypothesis is absent from belief snapshot: {hypothesis_id}")

    @property
    def snapshot_id(self) -> str:
        identity = (
            f"{self.case_id}|{self.asof_time.isoformat()}|{self.data_release_id}|"
            f"{self.vintage.value}|{self.evidence_hash}"
        )
        return hashlib.sha256(identity.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    case_id: str
    asof_time: datetime
    stage: ResearchStage
    evidence_strength: Decimal
    independent_clusters: tuple[str, ...]
    non_price_clusters: tuple[str, ...]
    validation_domains: tuple[ValidationDomain, ...]
    evidence_ids: tuple[str, ...]
    evidence_hash: str
    falsifier_seen: bool

    def __post_init__(self) -> None:
        if len(self.independent_clusters) != len(set(self.independent_clusters)):
            raise ValueError("assessment clusters must be unique")
        if not set(self.non_price_clusters) <= set(self.independent_clusters):
            raise ValueError("non-price clusters must be a subset of independent clusters")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("assessment evidence ids must be unique")
        object.__setattr__(self, "case_id", _require_text(self.case_id, field_name="case_id"))
        object.__setattr__(self, "asof_time", require_aware(self.asof_time, field_name="asof_time"))
        object.__setattr__(
            self,
            "evidence_strength",
            _require_probability(self.evidence_strength, field_name="evidence_strength"),
        )
        object.__setattr__(
            self,
            "evidence_hash",
            _require_sha256(self.evidence_hash, field_name="evidence_hash"),
        )
