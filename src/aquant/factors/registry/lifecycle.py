import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

from aquant.factors.evaluation import FactorApprovalStatus
from aquant.factors.evaluation.gates import GateDecision
from aquant.factors.exceptions import FactorRegistryError
from aquant.factors.lifecycle import feature_eligible, production_eligible, require_transition
from aquant.factors.mining import FactorCandidate
from aquant.factors.spec import FactorSpec, FactorStatus

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
    """Registry for versioned specs; legacy candidate records remain supported."""

    def __init__(self, database: Path | None = None) -> None:
        self._records: dict[tuple[str, str], RegisteredFactor] = {}
        self._specs: dict[tuple[str, str], FactorSpec] = {}
        self._expressions: dict[str, tuple[str, str]] = {}
        self._evidence: dict[tuple[str, str], str | None] = {}
        self._connection = sqlite3.connect(database) if database is not None else None
        if self._connection is not None:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS factor_specs (
                    factor_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    expression_hash TEXT NOT NULL UNIQUE,
                    spec_json TEXT NOT NULL,
                    evidence_hash TEXT,
                    PRIMARY KEY (factor_id, version)
                )
                """
            )
            self._load_specs()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()

    def __enter__(self) -> "FactorRegistry":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def register(self, record: RegisteredFactor | FactorSpec) -> None:
        if isinstance(record, FactorSpec):
            self._register_spec(record)
            return
        key = self._key(record.factor_id, record.version)
        if key in self._records or key in self._specs:
            raise ValueError(f"factor version already registered: {key}")
        self._records[key] = record

    def _register_spec(self, spec: FactorSpec) -> None:
        key = self._key(spec.factor_id, spec.version)
        if key in self._records or key in self._specs:
            raise FactorRegistryError(f"factor version already registered: {key}")
        duplicate = self._expressions.get(spec.expression_hash)
        if duplicate is not None:
            raise FactorRegistryError(f"canonical expression already registered by {duplicate}")
        self._specs[key] = spec
        self._expressions[spec.expression_hash] = key
        self._evidence[key] = None
        self._persist(spec, evidence_hash=None)

    def get(self, factor_id: str, version: str) -> FactorSpec:
        return self._specs[self._key(factor_id, version)]

    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(self._specs[key] for key in sorted(self._specs))

    def transition(
        self,
        factor_id: str,
        version: str,
        target: FactorApprovalStatus | FactorStatus,
        *,
        evidence_hash: str | None = None,
        gate_decision: GateDecision | None = None,
    ) -> RegisteredFactor | FactorSpec:
        key = self._key(factor_id, version)
        if key in self._specs:
            if not isinstance(target, FactorStatus):
                raise FactorRegistryError("spec lifecycle requires FactorStatus")
            spec = self._specs[key]
            require_transition(spec.status, target)
            prior_evidence = self._evidence[key]
            if target in {
                FactorStatus.PRODUCTION,
                FactorStatus.STANDALONE_PRODUCTION_ALPHA,
            } and (gate_decision is None or not gate_decision.passed):
                raise FactorRegistryError("production transition requires a passing gate decision")
            if gate_decision is not None:
                evidence_hash = gate_decision.evidence_hash
            if target in {
                FactorStatus.VALIDATED,
                FactorStatus.RESEARCH_VALIDATED,
                FactorStatus.FEATURE_ELIGIBLE,
                FactorStatus.STANDALONE_PRODUCTION_ALPHA,
                FactorStatus.APPROVED,
                FactorStatus.PAPER_TRADING,
                FactorStatus.PRODUCTION,
            } and not self._valid_hash(evidence_hash or prior_evidence):
                raise FactorRegistryError("validated/approved factors require an evidence SHA-256")
            updated_spec = spec.model_copy(update={"status": target})
            self._specs[key] = updated_spec
            self._evidence[key] = evidence_hash or prior_evidence
            self._persist(updated_spec, evidence_hash=self._evidence[key])
            return updated_spec
        if not isinstance(target, FactorApprovalStatus):
            raise ValueError("legacy lifecycle requires FactorApprovalStatus")
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

    def production_specs(self) -> tuple[FactorSpec, ...]:
        return tuple(spec for spec in self.specs() if production_eligible(spec.status))

    def feature_specs(self) -> tuple[FactorSpec, ...]:
        """Return the broad L2 pool available to L3, independent of strict production gates."""
        return tuple(spec for spec in self.specs() if feature_eligible(spec.status))

    def _persist(self, spec: FactorSpec, *, evidence_hash: str | None) -> None:
        if self._connection is None:
            return
        self._connection.execute(
            """
            INSERT INTO factor_specs (
                factor_id, version, expression_hash, spec_json, evidence_hash
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(factor_id, version) DO UPDATE SET
                expression_hash = excluded.expression_hash,
                spec_json = excluded.spec_json,
                evidence_hash = excluded.evidence_hash
            """,
            (
                spec.factor_id,
                spec.version,
                spec.expression_hash,
                spec.model_dump_json(),
                evidence_hash,
            ),
        )
        self._connection.commit()

    def _load_specs(self) -> None:
        assert self._connection is not None
        rows = self._connection.execute(
            "SELECT spec_json, evidence_hash FROM factor_specs ORDER BY factor_id, version"
        ).fetchall()
        for payload, evidence_hash in rows:
            spec = FactorSpec.model_validate_json(payload)
            key = (spec.factor_id, spec.version)
            self._specs[key] = spec
            self._expressions[spec.expression_hash] = key
            self._evidence[key] = evidence_hash

    @staticmethod
    def _valid_hash(value: str | None) -> bool:
        return (
            value is not None
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    @staticmethod
    def _key(factor_id: str, version: str) -> tuple[str, str]:
        key = (factor_id.strip(), version.strip())
        if not all(key):
            raise ValueError("factor id/version must not be blank")
        return key
