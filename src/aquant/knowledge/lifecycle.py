from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum

from aquant.domain.time import require_aware


class KnowledgeSourceKind(StrEnum):
    ACADEMIC_PAPER = "ACADEMIC_PAPER"
    BROKER_RESEARCH = "BROKER_RESEARCH"
    GITHUB_TOOL = "GITHUB_TOOL"
    REGULATION = "REGULATION"
    INVESTOR_PRACTICE = "INVESTOR_PRACTICE"
    OVERSEAS_RESEARCH = "OVERSEAS_RESEARCH"
    ALTERNATIVE_DATA = "ALTERNATIVE_DATA"
    FAILURE_CASE = "FAILURE_CASE"


class KnowledgeLifecycleStatus(StrEnum):
    DISCOVERED = "DISCOVERED"
    REPRODUCED = "REPRODUCED"
    VALIDATED = "VALIDATED"
    SHADOW = "SHADOW"
    APPROVED = "APPROVED"
    PRODUCTION = "PRODUCTION"
    REJECTED = "REJECTED"
    DEPRECATED = "DEPRECATED"


_TRANSITIONS: dict[KnowledgeLifecycleStatus, frozenset[KnowledgeLifecycleStatus]] = {
    KnowledgeLifecycleStatus.DISCOVERED: frozenset(
        {KnowledgeLifecycleStatus.REPRODUCED, KnowledgeLifecycleStatus.REJECTED}
    ),
    KnowledgeLifecycleStatus.REPRODUCED: frozenset(
        {KnowledgeLifecycleStatus.VALIDATED, KnowledgeLifecycleStatus.REJECTED}
    ),
    KnowledgeLifecycleStatus.VALIDATED: frozenset(
        {KnowledgeLifecycleStatus.SHADOW, KnowledgeLifecycleStatus.REJECTED}
    ),
    KnowledgeLifecycleStatus.SHADOW: frozenset(
        {KnowledgeLifecycleStatus.APPROVED, KnowledgeLifecycleStatus.REJECTED}
    ),
    KnowledgeLifecycleStatus.APPROVED: frozenset(
        {KnowledgeLifecycleStatus.PRODUCTION, KnowledgeLifecycleStatus.REJECTED}
    ),
    KnowledgeLifecycleStatus.PRODUCTION: frozenset({KnowledgeLifecycleStatus.DEPRECATED}),
    KnowledgeLifecycleStatus.REJECTED: frozenset(),
    KnowledgeLifecycleStatus.DEPRECATED: frozenset(),
}


def _valid_hash(value: str | None) -> bool:
    return (
        value is not None
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


@dataclass(frozen=True, slots=True)
class KnowledgeCandidate:
    candidate_id: str
    title: str
    source_kind: KnowledgeSourceKind
    source_uri: str
    hypothesis: str
    applicability: str
    acquired_at: datetime
    status: KnowledgeLifecycleStatus = KnowledgeLifecycleStatus.DISCOVERED
    evidence_hash: str | None = None
    failure_reason: str | None = None
    approved_by: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("candidate_id", "title", "source_uri", "hypothesis", "applicability"):
            value = getattr(self, field_name).strip()
            if not value:
                raise ValueError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self, "acquired_at", require_aware(self.acquired_at, field_name="acquired_at")
        )
        if self.evidence_hash is not None and not _valid_hash(self.evidence_hash):
            raise ValueError("knowledge evidence_hash must be SHA-256")
        if self.status not in {
            KnowledgeLifecycleStatus.DISCOVERED,
            KnowledgeLifecycleStatus.REJECTED,
        } and not _valid_hash(self.evidence_hash):
            raise ValueError("promoted knowledge requires an evidence SHA-256")
        if self.status is KnowledgeLifecycleStatus.REJECTED and not (
            self.failure_reason and self.failure_reason.strip()
        ):
            raise ValueError("rejected knowledge must enter the failure library")
        if self.status in {
            KnowledgeLifecycleStatus.APPROVED,
            KnowledgeLifecycleStatus.PRODUCTION,
        } and (not self.approved_by or not self.approved_by.strip()):
            raise ValueError("approved knowledge requires a human approver")


class KnowledgeLifecycleRegistry:
    """High-recall intake with progressively stricter evidence and human production gates."""

    def __init__(self) -> None:
        self._records: dict[str, KnowledgeCandidate] = {}

    def discover(self, candidate: KnowledgeCandidate) -> None:
        if candidate.status is not KnowledgeLifecycleStatus.DISCOVERED:
            raise ValueError("new knowledge candidates must start at DISCOVERED")
        if candidate.candidate_id in self._records:
            raise ValueError(f"knowledge candidate already exists: {candidate.candidate_id}")
        self._records[candidate.candidate_id] = candidate

    def get(self, candidate_id: str) -> KnowledgeCandidate:
        return self._records[candidate_id]

    def transition(
        self,
        candidate_id: str,
        target: KnowledgeLifecycleStatus,
        *,
        evidence_hash: str | None = None,
        failure_reason: str | None = None,
        approved_by: str | None = None,
        manual_confirmation: bool = False,
    ) -> KnowledgeCandidate:
        record = self._records[candidate_id]
        if target not in _TRANSITIONS[record.status]:
            raise ValueError(f"invalid knowledge lifecycle transition: {record.status} -> {target}")

        resolved_hash = evidence_hash or record.evidence_hash
        if target in {
            KnowledgeLifecycleStatus.REPRODUCED,
            KnowledgeLifecycleStatus.VALIDATED,
            KnowledgeLifecycleStatus.SHADOW,
            KnowledgeLifecycleStatus.APPROVED,
            KnowledgeLifecycleStatus.PRODUCTION,
        } and not _valid_hash(resolved_hash):
            raise ValueError("knowledge promotion requires a reproducible evidence SHA-256")
        if target is KnowledgeLifecycleStatus.REJECTED and not (
            failure_reason and failure_reason.strip()
        ):
            raise ValueError("rejected knowledge requires a failure reason")

        resolved_approver = approved_by or record.approved_by
        if target is KnowledgeLifecycleStatus.APPROVED and (
            not manual_confirmation or not resolved_approver or not resolved_approver.strip()
        ):
            raise ValueError("knowledge approval requires explicit human confirmation")
        if target is KnowledgeLifecycleStatus.PRODUCTION and not record.approved_by:
            raise ValueError("production knowledge must inherit a prior human approval")

        updated = replace(
            record,
            status=target,
            evidence_hash=resolved_hash,
            failure_reason=failure_reason.strip() if failure_reason else record.failure_reason,
            approved_by=resolved_approver.strip() if resolved_approver else None,
        )
        self._records[candidate_id] = updated
        return updated

    def failure_library(self) -> tuple[KnowledgeCandidate, ...]:
        return tuple(
            sorted(
                (
                    record
                    for record in self._records.values()
                    if record.status is KnowledgeLifecycleStatus.REJECTED
                    or record.source_kind is KnowledgeSourceKind.FAILURE_CASE
                ),
                key=lambda record: record.candidate_id,
            )
        )
