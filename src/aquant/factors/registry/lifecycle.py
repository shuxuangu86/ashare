from dataclasses import dataclass, replace

from aquant.factors.evaluation import FactorApprovalStatus
from aquant.factors.mining import FactorCandidate

_TRANSITIONS: dict[FactorApprovalStatus, frozenset[FactorApprovalStatus]] = {
    FactorApprovalStatus.DRAFT: frozenset({FactorApprovalStatus.COMPUTED}),
    FactorApprovalStatus.COMPUTED: frozenset(
        {FactorApprovalStatus.VALIDATED, FactorApprovalStatus.DEPRECATED}
    ),
    FactorApprovalStatus.VALIDATED: frozenset(
        {FactorApprovalStatus.APPROVED, FactorApprovalStatus.DEPRECATED}
    ),
    FactorApprovalStatus.APPROVED: frozenset(
        {FactorApprovalStatus.PRODUCTION, FactorApprovalStatus.DEPRECATED}
    ),
    FactorApprovalStatus.PRODUCTION: frozenset({FactorApprovalStatus.DEPRECATED}),
    FactorApprovalStatus.DEPRECATED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class RegisteredFactor:
    factor_id: str
    version: str
    candidate: FactorCandidate
    status: FactorApprovalStatus = FactorApprovalStatus.DRAFT
    evidence_hash: str | None = None


class FactorRegistry:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], RegisteredFactor] = {}

    def register(self, record: RegisteredFactor) -> None:
        key = self._key(record.factor_id, record.version)
        if key in self._records:
            raise ValueError(f"factor version already registered: {key}")
        self._records[key] = record

    def transition(
        self,
        factor_id: str,
        version: str,
        target: FactorApprovalStatus,
        *,
        evidence_hash: str | None = None,
    ) -> RegisteredFactor:
        key = self._key(factor_id, version)
        record = self._records[key]
        if target not in _TRANSITIONS[record.status]:
            raise ValueError(f"invalid factor lifecycle transition: {record.status} -> {target}")
        if target in {FactorApprovalStatus.VALIDATED, FactorApprovalStatus.APPROVED} and (
            evidence_hash is None or len(evidence_hash) != 64
        ):
            raise ValueError("validated/approved factors require an evidence SHA-256")
        updated = replace(
            record, status=target, evidence_hash=evidence_hash or record.evidence_hash
        )
        self._records[key] = updated
        return updated

    def production_eligible(self) -> tuple[RegisteredFactor, ...]:
        return tuple(
            record
            for record in self._records.values()
            if record.status in {FactorApprovalStatus.APPROVED, FactorApprovalStatus.PRODUCTION}
        )

    @staticmethod
    def _key(factor_id: str, version: str) -> tuple[str, str]:
        key = (factor_id.strip(), version.strip())
        if not all(key):
            raise ValueError("factor id/version must not be blank")
        return key
