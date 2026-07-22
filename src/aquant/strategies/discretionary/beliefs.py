import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from aquant.domain.data_release import DataReleaseId
from aquant.domain.time import require_aware
from aquant.strategies.discretionary.calibration import CalibrationLedger
from aquant.strategies.discretionary.models import (
    CORE_VALIDATION_DOMAINS,
    BeliefSnapshot,
    EstimateVintage,
    EvidenceAssessment,
    EvidenceFamily,
    EvidenceMaturity,
    EvidenceObservation,
    HypothesisProbability,
    ResearchCase,
    ResearchStage,
)


@dataclass(frozen=True, slots=True)
class BeliefUpdate:
    snapshot: BeliefSnapshot
    assessment: EvidenceAssessment


class BayesianBeliefEngine:
    """PIT Bayesian updater that conservatively deduplicates dependent evidence."""

    def __init__(self, *, minimum_non_price_clusters: int = 2) -> None:
        if minimum_non_price_clusters < 2:
            raise ValueError("position-grade evidence requires at least two non-price clusters")
        self._minimum_non_price_clusters = minimum_non_price_clusters

    def update(
        self,
        case: ResearchCase,
        observations: tuple[EvidenceObservation, ...],
        *,
        asof_time: datetime,
        data_release_id: DataReleaseId,
        calibration: CalibrationLedger | None = None,
        vintage: EstimateVintage = EstimateVintage.REAL_TIME,
    ) -> BeliefUpdate:
        normalized_asof = require_aware(asof_time, field_name="asof_time")
        if normalized_asof < case.created_at:
            raise ValueError("belief snapshot cannot predate its research case")
        self._validate_observations(case, observations)
        visible = tuple(
            sorted(
                (
                    observation
                    for observation in observations
                    if observation.available_at <= normalized_asof
                ),
                key=lambda item: (item.available_at, item.evidence_id),
            )
        )
        effective_quality = {
            observation.evidence_id: self._effective_quality(
                observation, normalized_asof, calibration
            )
            for observation in visible
        }
        effective = tuple(
            observation for observation in visible if effective_quality[observation.evidence_id] > 0
        )

        hypothesis_ids = tuple(hypothesis.hypothesis_id for hypothesis in case.hypotheses)
        log_scores = {
            hypothesis.hypothesis_id: hypothesis.prior_probability.ln()
            for hypothesis in case.hypotheses
        }
        by_cluster: dict[str, list[EvidenceObservation]] = defaultdict(list)
        for observation in effective:
            by_cluster[observation.dependency_cluster_id].append(observation)

        for cluster_observations in by_cluster.values():
            cluster_weight = sum(
                (
                    effective_quality[observation.evidence_id]
                    for observation in cluster_observations
                ),
                Decimal("0"),
            )
            if cluster_weight == 0:
                continue
            cluster_confidence = max(
                effective_quality[observation.evidence_id] for observation in cluster_observations
            )
            for hypothesis_id in hypothesis_ids:
                weighted_log_likelihood = sum(
                    (
                        self._likelihood_for(observation, hypothesis_id).ln()
                        * effective_quality[observation.evidence_id]
                        for observation in cluster_observations
                    ),
                    Decimal("0"),
                )
                # Averaging within a dependency cluster prevents duplicated articles or
                # mutually dependent channel checks from being counted as independent proof.
                log_scores[hypothesis_id] += (
                    weighted_log_likelihood / cluster_weight * cluster_confidence
                )

        maximum_log_score = max(log_scores.values())
        unnormalized = {
            hypothesis_id: (score - maximum_log_score).exp()
            for hypothesis_id, score in log_scores.items()
        }
        normalizer = sum(unnormalized.values(), Decimal("0"))
        resolved_probabilities: list[HypothesisProbability] = []
        allocated = Decimal("0")
        for index, hypothesis_id in enumerate(hypothesis_ids):
            if index == len(hypothesis_ids) - 1:
                probability = Decimal("1") - allocated
            else:
                probability = unnormalized[hypothesis_id] / normalizer
                allocated += probability
            resolved_probabilities.append(HypothesisProbability(hypothesis_id, probability))

        evidence_hash = self._evidence_hash(case, visible, data_release_id)
        evidence_ids = tuple(observation.evidence_id for observation in visible)
        snapshot = BeliefSnapshot(
            case.case_id,
            normalized_asof,
            data_release_id,
            vintage,
            tuple(resolved_probabilities),
            evidence_ids,
            evidence_hash,
        )
        assessment = self._assess(
            case,
            effective,
            effective_quality,
            normalized_asof,
            evidence_ids,
            evidence_hash,
            self._minimum_non_price_clusters,
        )
        return BeliefUpdate(snapshot, assessment)

    @staticmethod
    def _validate_observations(
        case: ResearchCase, observations: tuple[EvidenceObservation, ...]
    ) -> None:
        evidence_ids = [observation.evidence_id for observation in observations]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("belief update cannot contain duplicate evidence ids")
        variable_ids = {variable.variable_id for variable in case.key_variables}
        hypothesis_ids = {hypothesis.hypothesis_id for hypothesis in case.hypotheses}
        for observation in observations:
            if observation.case_id != case.case_id:
                raise ValueError("evidence belongs to a different research case")
            if observation.variable_id not in variable_ids:
                raise ValueError("evidence references an undefined key variable")
            observation_hypotheses = {
                likelihood.hypothesis_id for likelihood in observation.likelihoods
            }
            if observation_hypotheses != hypothesis_ids:
                raise ValueError("evidence must score every competing hypothesis")

    @staticmethod
    def _likelihood_for(observation: EvidenceObservation, hypothesis_id: str) -> Decimal:
        return next(
            likelihood.likelihood_ratio
            for likelihood in observation.likelihoods
            if likelihood.hypothesis_id == hypothesis_id
        )

    @staticmethod
    def _effective_quality(
        observation: EvidenceObservation,
        asof_time: datetime,
        calibration: CalibrationLedger | None,
    ) -> Decimal:
        reliability = observation.quality.reliability
        if calibration is not None:
            reliability = calibration.reliability(
                source_id=observation.source_id,
                method_version=observation.method_version,
                asof_time=asof_time,
                default_reliability=reliability,
            )
        return observation.quality.combined(reliability=reliability)

    @staticmethod
    def _evidence_hash(
        case: ResearchCase,
        observations: tuple[EvidenceObservation, ...],
        data_release_id: DataReleaseId,
    ) -> str:
        payload = {
            "case_id": case.case_id,
            "data_release_id": str(data_release_id),
            "evidence": [
                {
                    "available_at": observation.available_at.isoformat(),
                    "evidence_id": observation.evidence_id,
                    "method_version": observation.method_version,
                    "raw_content_sha256": observation.raw_content_sha256,
                }
                for observation in observations
            ],
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _assess(
        case: ResearchCase,
        observations: tuple[EvidenceObservation, ...],
        effective_quality: dict[str, Decimal],
        asof_time: datetime,
        evidence_ids: tuple[str, ...],
        evidence_hash: str,
        minimum_non_price_clusters: int,
    ) -> EvidenceAssessment:
        clusters = tuple(sorted({item.dependency_cluster_id for item in observations}))
        non_price_clusters = tuple(
            sorted(
                {
                    item.dependency_cluster_id
                    for item in observations
                    if item.family is not EvidenceFamily.PRICE
                }
            )
        )
        validation_domains = tuple(
            sorted(
                {domain for item in observations for domain in item.validation_domains},
                key=lambda domain: domain.value,
            )
        )
        cluster_strengths = tuple(
            max(
                effective_quality[item.evidence_id]
                for item in observations
                if item.dependency_cluster_id == cluster_id
            )
            for cluster_id in non_price_clusters
        )
        if cluster_strengths:
            average_strength = sum(cluster_strengths, Decimal("0")) / Decimal(
                len(cluster_strengths)
            )
            breadth = min(Decimal("1"), Decimal(len(cluster_strengths)) / Decimal("4"))
            evidence_strength = average_strength * breadth
        else:
            evidence_strength = Decimal("0")

        falsifier_seen = any(item.falsifies_fundamental for item in observations)
        if falsifier_seen:
            stage = ResearchStage.INVALIDATED
        elif not observations:
            stage = ResearchStage.RADAR
        elif not non_price_clusters:
            stage = ResearchStage.HYPOTHESIS
        elif len(non_price_clusters) < minimum_non_price_clusters:
            stage = ResearchStage.WATCHLIST
        elif set(validation_domains) >= CORE_VALIDATION_DOMAINS and any(
            item.maturity is EvidenceMaturity.FINANCIAL_CONFIRMED for item in observations
        ):
            stage = ResearchStage.CORE
        elif any(
            item.maturity
            in {EvidenceMaturity.OPERATING_CONFIRMED, EvidenceMaturity.FINANCIAL_CONFIRMED}
            for item in observations
        ):
            stage = ResearchStage.EVIDENCE
        else:
            stage = ResearchStage.OBSERVATION

        return EvidenceAssessment(
            case.case_id,
            asof_time,
            stage,
            evidence_strength,
            clusters,
            non_price_clusters,
            validation_domains,
            evidence_ids,
            evidence_hash,
            falsifier_seen,
        )
