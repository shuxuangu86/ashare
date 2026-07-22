from dataclasses import dataclass
from datetime import datetime

from aquant.config.settings import DiscretionaryStrategySettings
from aquant.domain.data_release import DataReleaseId
from aquant.strategies.discretionary.beliefs import BayesianBeliefEngine, BeliefUpdate
from aquant.strategies.discretionary.calibration import CalibrationLedger
from aquant.strategies.discretionary.models import (
    EstimateVintage,
    EvidenceObservation,
    ResearchCase,
)
from aquant.strategies.discretionary.positioning import (
    AllocationProposal,
    MarketExpectationSnapshot,
    PayoffProfile,
    PositionPolicy,
    PositionSizingEngine,
    PriorPositionState,
)


@dataclass(frozen=True, slots=True)
class DiscretionaryDecision:
    belief: BeliefUpdate
    proposal: AllocationProposal


class DiscretionaryFundamentalStrategy:
    """Research-side facade; it never approves allocations or submits orders."""

    def __init__(
        self,
        *,
        policy: PositionPolicy | None = None,
        calibration: CalibrationLedger | None = None,
        minimum_non_price_clusters: int = 2,
    ) -> None:
        self._belief_engine = BayesianBeliefEngine(
            minimum_non_price_clusters=minimum_non_price_clusters
        )
        self._position_engine = PositionSizingEngine(policy)
        self._calibration = calibration

    @classmethod
    def from_settings(
        cls, settings: DiscretionaryStrategySettings
    ) -> "DiscretionaryFundamentalStrategy":
        if not settings.enabled:
            raise ValueError("discretionary strategy is disabled by configuration")
        policy = PositionPolicy(
            observation_cap=settings.observation_weight_cap,
            evidence_cap=settings.evidence_weight_cap,
            core_cap=settings.core_weight_cap,
            minimum_probability_edge=settings.minimum_probability_edge,
            minimum_evidence_upgrade=settings.minimum_evidence_upgrade,
            correlation_floor=settings.correlation_floor,
            sizing_multiplier=settings.sizing_multiplier,
            require_human_approval=settings.require_human_approval,
        )
        return cls(
            policy=policy,
            calibration=CalibrationLedger(prior_weight=settings.calibration_prior_weight),
            minimum_non_price_clusters=settings.minimum_independent_non_price_clusters,
        )

    def review(
        self,
        case: ResearchCase,
        observations: tuple[EvidenceObservation, ...],
        expectation: MarketExpectationSnapshot,
        payoff: PayoffProfile,
        *,
        asof_time: datetime,
        data_release_id: DataReleaseId,
        prior: PriorPositionState | None = None,
    ) -> DiscretionaryDecision:
        belief = self._belief_engine.update(
            case,
            observations,
            asof_time=asof_time,
            data_release_id=data_release_id,
            calibration=self._calibration,
            vintage=EstimateVintage.REAL_TIME,
        )
        proposal = self._position_engine.propose(
            case,
            belief,
            expectation,
            payoff,
            prior=prior,
        )
        return DiscretionaryDecision(belief, proposal)
