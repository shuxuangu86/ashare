import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_DOWN, Decimal
from enum import StrEnum

from aquant.domain.data_release import DataReleaseId
from aquant.domain.identifiers import Symbol
from aquant.domain.portfolio import TargetPortfolio, TargetPosition
from aquant.domain.time import require_aware
from aquant.strategies.discretionary.beliefs import BeliefUpdate
from aquant.strategies.discretionary.models import (
    EstimateVintage,
    ResearchCase,
    ResearchStage,
    ValidationDomain,
)


class AllocationTier(StrEnum):
    NONE = "NONE"
    OBSERVATION = "OBSERVATION"
    EVIDENCE = "EVIDENCE"
    CORE = "CORE"


class AllocationAction(StrEnum):
    WATCH = "WATCH"
    OPEN = "OPEN"
    ADD = "ADD"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"


_STAGE_RANK = {
    ResearchStage.RADAR: 0,
    ResearchStage.HYPOTHESIS: 1,
    ResearchStage.WATCHLIST: 2,
    ResearchStage.OBSERVATION: 3,
    ResearchStage.EVIDENCE: 4,
    ResearchStage.CORE: 5,
    ResearchStage.INVALIDATED: -1,
}


def _probability(value: Decimal, *, field_name: str) -> Decimal:
    normalized = Decimal(value)
    if not Decimal("0") <= normalized <= Decimal("1"):
        raise ValueError(f"{field_name} must be between 0 and 1")
    return normalized


def _sha256(value: str, *, field_name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    return normalized


@dataclass(frozen=True, slots=True)
class MarketExpectationSnapshot:
    case_id: str
    fundamental_hypothesis_id: str
    implied_probability: Decimal
    current_price: Decimal
    rationale: str
    source_uri: str
    observed_at: datetime
    available_at: datetime

    def __post_init__(self) -> None:
        observed_at = require_aware(self.observed_at, field_name="observed_at")
        available_at = require_aware(self.available_at, field_name="available_at")
        if available_at < observed_at:
            raise ValueError("market expectation available_at cannot precede observed_at")
        if Decimal(self.current_price) <= 0:
            raise ValueError("market expectation current_price must be positive")
        for field_name in ("case_id", "fundamental_hypothesis_id", "rationale", "source_uri"):
            value = getattr(self, field_name).strip()
            if not value:
                raise ValueError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self,
            "implied_probability",
            _probability(self.implied_probability, field_name="implied_probability"),
        )
        object.__setattr__(self, "current_price", Decimal(self.current_price))
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "available_at", available_at)

    @property
    def expectation_id(self) -> str:
        payload = {
            "available_at": self.available_at.isoformat(),
            "case_id": self.case_id,
            "current_price": str(self.current_price),
            "fundamental_hypothesis_id": self.fundamental_hypothesis_id,
            "implied_probability": str(self.implied_probability),
            "rationale": self.rationale,
            "source_uri": self.source_uri,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class PayoffProfile:
    upside_return: Decimal
    failure_loss: Decimal
    variable_correlation: Decimal

    def __post_init__(self) -> None:
        upside = Decimal(self.upside_return)
        failure_loss = Decimal(self.failure_loss)
        correlation = Decimal(self.variable_correlation)
        if upside <= 0 or failure_loss <= 0:
            raise ValueError("payoff upside_return and failure_loss must be positive")
        if not Decimal("0") <= correlation <= Decimal("1"):
            raise ValueError("variable_correlation must be between 0 and 1")
        object.__setattr__(self, "upside_return", upside)
        object.__setattr__(self, "failure_loss", failure_loss)
        object.__setattr__(self, "variable_correlation", correlation)


@dataclass(frozen=True, slots=True)
class PositionPolicy:
    observation_cap: Decimal = Decimal("0.005")
    evidence_cap: Decimal = Decimal("0.02")
    core_cap: Decimal = Decimal("0.05")
    minimum_probability_edge: Decimal = Decimal("0.05")
    minimum_evidence_upgrade: Decimal = Decimal("0.05")
    correlation_floor: Decimal = Decimal("0.25")
    sizing_multiplier: Decimal = Decimal("0.05")
    require_human_approval: bool = True

    def __post_init__(self) -> None:
        observation = _probability(self.observation_cap, field_name="observation_cap")
        evidence = _probability(self.evidence_cap, field_name="evidence_cap")
        core = _probability(self.core_cap, field_name="core_cap")
        if not Decimal("0") < observation <= evidence <= core:
            raise ValueError(
                "position caps must be positive and ordered observation <= evidence <= core"
            )
        for field_name in (
            "minimum_probability_edge",
            "minimum_evidence_upgrade",
            "correlation_floor",
        ):
            object.__setattr__(
                self, field_name, _probability(getattr(self, field_name), field_name=field_name)
            )
        if Decimal(self.sizing_multiplier) <= 0:
            raise ValueError("sizing_multiplier must be positive")
        if not self.require_human_approval:
            raise ValueError("discretionary allocations always require human approval")
        object.__setattr__(self, "observation_cap", observation)
        object.__setattr__(self, "evidence_cap", evidence)
        object.__setattr__(self, "core_cap", core)
        object.__setattr__(self, "sizing_multiplier", Decimal(self.sizing_multiplier))

    @property
    def policy_hash(self) -> str:
        payload = {
            "core_cap": str(self.core_cap),
            "correlation_floor": str(self.correlation_floor),
            "evidence_cap": str(self.evidence_cap),
            "minimum_evidence_upgrade": str(self.minimum_evidence_upgrade),
            "minimum_probability_edge": str(self.minimum_probability_edge),
            "observation_cap": str(self.observation_cap),
            "require_human_approval": self.require_human_approval,
            "sizing_multiplier": str(self.sizing_multiplier),
        }
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class PriorPositionState:
    current_weight: Decimal
    stage: ResearchStage
    evidence_strength: Decimal
    evidence_hash: str
    validation_domains: tuple[ValidationDomain, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_weight",
            _probability(self.current_weight, field_name="current_weight"),
        )
        object.__setattr__(
            self,
            "evidence_strength",
            _probability(self.evidence_strength, field_name="evidence_strength"),
        )
        object.__setattr__(
            self, "evidence_hash", _sha256(self.evidence_hash, field_name="evidence_hash")
        )
        if len(self.validation_domains) != len(set(self.validation_domains)):
            raise ValueError("prior validation domains must be unique")


@dataclass(frozen=True, slots=True)
class AllocationProposal:
    proposal_id: str
    case_id: str
    symbol: Symbol
    asof_time: datetime
    data_release_id: DataReleaseId
    belief_snapshot_id: str
    market_expectation_id: str
    payoff_profile: PayoffProfile
    policy_hash: str
    action: AllocationAction
    tier: AllocationTier
    target_weight: Decimal
    current_weight: Decimal
    posterior_probability: Decimal
    market_probability: Decimal
    expected_return: Decimal
    evidence_strength: Decimal
    evidence_hash: str
    reasons: tuple[str, ...]
    requires_manual_approval: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "target_weight",
            "current_weight",
            "posterior_probability",
            "market_probability",
            "evidence_strength",
        ):
            object.__setattr__(
                self, field_name, _probability(getattr(self, field_name), field_name=field_name)
            )
        object.__setattr__(self, "proposal_id", _sha256(self.proposal_id, field_name="proposal_id"))
        for field_name in ("belief_snapshot_id", "market_expectation_id", "policy_hash"):
            object.__setattr__(
                self, field_name, _sha256(getattr(self, field_name), field_name=field_name)
            )
        if not self.case_id.strip():
            raise ValueError("allocation proposal case_id must not be blank")
        if not self.reasons or any(not reason.strip() for reason in self.reasons):
            raise ValueError("allocation proposal requires non-blank reasons")
        if not self.requires_manual_approval:
            raise ValueError("allocation proposal cannot bypass manual approval")
        object.__setattr__(self, "case_id", self.case_id.strip())
        object.__setattr__(self, "asof_time", require_aware(self.asof_time, field_name="asof_time"))
        object.__setattr__(self, "expected_return", Decimal(self.expected_return))
        object.__setattr__(
            self, "evidence_hash", _sha256(self.evidence_hash, field_name="evidence_hash")
        )
        object.__setattr__(self, "reasons", tuple(reason.strip() for reason in self.reasons))


@dataclass(frozen=True, slots=True)
class AllocationApproval:
    approval_id: str
    proposal: AllocationProposal
    approved_weight: Decimal
    approved_by: str
    rationale: str
    approved_at: datetime
    manual_confirmation: bool

    def __post_init__(self) -> None:
        approved_at = require_aware(self.approved_at, field_name="approved_at")
        weight = _probability(self.approved_weight, field_name="approved_weight")
        if approved_at < self.proposal.asof_time:
            raise ValueError("allocation approval cannot predate its proposal")
        if weight > self.proposal.target_weight:
            raise ValueError("approved weight cannot exceed the proposed weight")
        if not self.manual_confirmation:
            raise ValueError("allocation approval requires explicit manual confirmation")
        for field_name in ("approval_id", "approved_by", "rationale"):
            value = getattr(self, field_name).strip()
            if not value:
                raise ValueError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "approved_weight", weight)
        object.__setattr__(self, "approved_at", approved_at)


class PositionSizingEngine:
    def __init__(self, policy: PositionPolicy | None = None) -> None:
        self._policy = policy or PositionPolicy()

    def propose(
        self,
        case: ResearchCase,
        belief: BeliefUpdate,
        expectation: MarketExpectationSnapshot,
        payoff: PayoffProfile,
        *,
        prior: PriorPositionState | None = None,
    ) -> AllocationProposal:
        snapshot = belief.snapshot
        assessment = belief.assessment
        if snapshot.vintage is not EstimateVintage.REAL_TIME:
            raise ValueError("retrospective estimates cannot generate allocation proposals")
        if snapshot.case_id != case.case_id or assessment.case_id != case.case_id:
            raise ValueError("belief update belongs to a different research case")
        if snapshot.evidence_hash != assessment.evidence_hash:
            raise ValueError("belief snapshot and evidence assessment lineage do not match")
        if expectation.case_id != case.case_id:
            raise ValueError("market expectation belongs to a different research case")
        if expectation.fundamental_hypothesis_id != case.fundamental_hypothesis_id:
            raise ValueError("market expectation targets the wrong fundamental hypothesis")
        if expectation.available_at > snapshot.asof_time:
            raise ValueError("future market expectations cannot enter a decision")

        current_weight = prior.current_weight if prior is not None else Decimal("0")
        posterior = snapshot.probability_for(case.fundamental_hypothesis_id)
        probability_edge = posterior - expectation.implied_probability
        expected_return = (
            posterior * payoff.upside_return - (Decimal("1") - posterior) * payoff.failure_loss
        )
        cap, tier = self._cap_for_stage(assessment.stage)
        reasons = [f"RESEARCH_STAGE_{assessment.stage.value}"]

        if assessment.stage is ResearchStage.INVALIDATED:
            target_weight = Decimal("0")
            reasons.append("FUNDAMENTAL_HYPOTHESIS_INVALIDATED")
        elif cap == 0:
            target_weight = Decimal("0")
            reasons.append("EVIDENCE_NOT_POSITION_ELIGIBLE")
        elif probability_edge < self._policy.minimum_probability_edge:
            target_weight = Decimal("0")
            reasons.append("INSUFFICIENT_MARKET_EXPECTATION_GAP")
        elif expected_return <= 0:
            target_weight = Decimal("0")
            reasons.append("NON_POSITIVE_EXPECTED_RETURN")
        else:
            denominator = payoff.failure_loss * max(
                payoff.variable_correlation, self._policy.correlation_floor
            )
            raw_weight = (
                self._policy.sizing_multiplier
                * assessment.evidence_strength
                * expected_return
                / denominator
            )
            target_weight = min(cap, raw_weight)
            reasons.append("POSITIVE_EXPECTATION_ADJUSTED_ASYMMETRY")

        if prior is not None and target_weight > current_weight:
            evidence_upgraded = self._evidence_upgraded(prior, belief)
            if not evidence_upgraded:
                target_weight = current_weight
                reasons.append("ADD_BLOCKED_WITHOUT_EVIDENCE_UPGRADE")
        target_weight = target_weight.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
        action = self._action(current_weight, target_weight)
        reasons.append("PENDING_HUMAN_APPROVAL")
        proposal_id = self._proposal_id(
            case,
            snapshot.asof_time,
            snapshot.data_release_id,
            snapshot.snapshot_id,
            expectation.expectation_id,
            payoff,
            self._policy.policy_hash,
            action,
            target_weight,
            snapshot.evidence_hash,
        )
        return AllocationProposal(
            proposal_id,
            case.case_id,
            case.symbol,
            snapshot.asof_time,
            snapshot.data_release_id,
            snapshot.snapshot_id,
            expectation.expectation_id,
            payoff,
            self._policy.policy_hash,
            action,
            tier,
            target_weight,
            current_weight,
            posterior,
            expectation.implied_probability,
            expected_return,
            assessment.evidence_strength,
            snapshot.evidence_hash,
            tuple(reasons),
        )

    def _cap_for_stage(self, stage: ResearchStage) -> tuple[Decimal, AllocationTier]:
        if stage is ResearchStage.OBSERVATION:
            return self._policy.observation_cap, AllocationTier.OBSERVATION
        if stage is ResearchStage.EVIDENCE:
            return self._policy.evidence_cap, AllocationTier.EVIDENCE
        if stage is ResearchStage.CORE:
            return self._policy.core_cap, AllocationTier.CORE
        return Decimal("0"), AllocationTier.NONE

    def _evidence_upgraded(self, prior: PriorPositionState, belief: BeliefUpdate) -> bool:
        assessment = belief.assessment
        if assessment.evidence_hash == prior.evidence_hash:
            return False
        if _STAGE_RANK[assessment.stage] < _STAGE_RANK[prior.stage]:
            return False
        stage_improved = _STAGE_RANK[assessment.stage] > _STAGE_RANK[prior.stage]
        strength_improved = (
            assessment.evidence_strength
            >= prior.evidence_strength + self._policy.minimum_evidence_upgrade
        )
        domains_improved = set(assessment.validation_domains) > set(prior.validation_domains)
        return stage_improved or strength_improved or domains_improved

    @staticmethod
    def _action(current_weight: Decimal, target_weight: Decimal) -> AllocationAction:
        if current_weight == target_weight:
            return AllocationAction.HOLD if current_weight > 0 else AllocationAction.WATCH
        if current_weight == 0:
            return AllocationAction.OPEN
        if target_weight == 0:
            return AllocationAction.EXIT
        if target_weight > current_weight:
            return AllocationAction.ADD
        return AllocationAction.REDUCE

    @staticmethod
    def _proposal_id(
        case: ResearchCase,
        asof_time: datetime,
        data_release_id: DataReleaseId,
        belief_snapshot_id: str,
        market_expectation_id: str,
        payoff: PayoffProfile,
        policy_hash: str,
        action: AllocationAction,
        target_weight: Decimal,
        evidence_hash: str,
    ) -> str:
        payload = {
            "action": action.value,
            "asof_time": asof_time.isoformat(),
            "case_id": case.case_id,
            "data_release_id": str(data_release_id),
            "belief_snapshot_id": belief_snapshot_id,
            "evidence_hash": evidence_hash,
            "market_expectation_id": market_expectation_id,
            "payoff": {
                "failure_loss": str(payoff.failure_loss),
                "upside_return": str(payoff.upside_return),
                "variable_correlation": str(payoff.variable_correlation),
            },
            "policy_hash": policy_hash,
            "symbol": str(case.symbol),
            "target_weight": str(target_weight),
        }
        canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


def approve_allocation(
    proposal: AllocationProposal,
    *,
    approved_by: str,
    rationale: str,
    approved_at: datetime,
    manual_confirmation: bool,
    approved_weight: Decimal | None = None,
) -> AllocationApproval:
    resolved_weight = (
        proposal.target_weight if approved_weight is None else Decimal(approved_weight)
    )
    payload = (
        f"{proposal.proposal_id}|{approved_by.strip()}|{approved_at.isoformat()}|"
        f"{resolved_weight}|{rationale.strip()}"
    )
    approval_id = hashlib.sha256(payload.encode()).hexdigest()
    return AllocationApproval(
        approval_id,
        proposal,
        resolved_weight,
        approved_by,
        rationale,
        approved_at,
        manual_confirmation,
    )


def build_approved_target_portfolio(
    approvals: tuple[AllocationApproval, ...],
    *,
    held_symbols: tuple[Symbol, ...],
    strategy_id: str,
    strategy_version: str,
    trade_date: date,
    asof_time: datetime,
    data_release_id: DataReleaseId,
) -> TargetPortfolio:
    normalized_asof = require_aware(asof_time, field_name="asof_time")
    approval_symbols = [approval.proposal.symbol for approval in approvals]
    if len(approval_symbols) != len(set(approval_symbols)):
        raise ValueError("approved discretionary book cannot contain duplicate symbols")
    if not set(held_symbols) <= set(approval_symbols):
        raise ValueError("every existing discretionary holding must receive a fresh manual review")
    for approval in approvals:
        if approval.approved_at > normalized_asof:
            raise ValueError("future approvals cannot enter a target portfolio")
        if approval.proposal.data_release_id != data_release_id:
            raise ValueError("allocation approval uses a different data release")
        if not approval.manual_confirmation:
            raise ValueError("target portfolio requires manually confirmed approvals")

    approval_bundle = hashlib.sha256(
        "|".join(sorted(approval.approval_id for approval in approvals)).encode()
    ).hexdigest()[:16]
    positions = tuple(
        TargetPosition(approval.proposal.symbol, approval.approved_weight)
        for approval in sorted(approvals, key=lambda item: item.proposal.symbol)
        if approval.approved_weight > 0
    )
    return TargetPortfolio(
        strategy_id,
        trade_date,
        normalized_asof,
        str(data_release_id),
        f"{strategy_version}:{approval_bundle}",
        positions,
    )
