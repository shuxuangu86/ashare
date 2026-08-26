import hashlib
import json
from datetime import UTC, date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aquant.domain.data_release import DataReleaseId


class FeatureSetStatus(StrEnum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"
    DEPRECATED = "DEPRECATED"
    PRODUCTION = "PRODUCTION"


class FeatureRole(StrEnum):
    ALPHA_CANDIDATE = "ALPHA_CANDIDATE"
    RISK_CONTROL = "RISK_CONTROL"
    CONTROL_FEATURE = "CONTROL_FEATURE"


class FactorMember(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    factor_id: str
    factor_version: str
    role: FeatureRole = FeatureRole.ALPHA_CANDIDATE

    @field_validator("factor_id", "factor_version")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("feature-set factor identity must not be blank")
        return resolved


class FeatureSetSpec(BaseModel):
    """Versioned feature membership; every member pins a factor version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_set_id: str
    version: str
    description: str
    target_horizon: int
    universe: str
    factor_members: tuple[FactorMember, ...]
    preprocessing: tuple[str, ...] = ()
    neutralization: tuple[str, ...] = ()
    selection_method: str
    created_from_experiment: str
    data_release_id: DataReleaseId
    status: FeatureSetStatus = FeatureSetStatus.DRAFT
    standardization_method: str = "none"
    winsorization_method: str = "none"
    missing_value_strategy: str = "preserve"
    effective_from: date | None = None
    training_window_days: int | None = None
    evaluation_window: tuple[date, date] | None = None
    code_version: str = "working-tree"
    config_hash: str = "0" * 64
    lineage: tuple[tuple[str, str], ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content_hash: str = ""

    @field_validator(
        "feature_set_id",
        "version",
        "description",
        "universe",
        "selection_method",
        "created_from_experiment",
        "standardization_method",
        "winsorization_method",
        "missing_value_strategy",
        "code_version",
    )
    @classmethod
    def _not_blank(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("feature-set metadata must not be blank")
        return resolved

    @model_validator(mode="after")
    def _validate_members(self) -> "FeatureSetSpec":
        if self.target_horizon <= 0:
            raise ValueError("feature-set target horizon must be positive")
        if not self.factor_members:
            raise ValueError("feature set must contain at least one factor")
        keys = [(member.factor_id, member.factor_version) for member in self.factor_members]
        if len(keys) != len(set(keys)):
            raise ValueError("feature set must not contain duplicate factor versions")
        if self.training_window_days is not None and self.training_window_days <= 0:
            raise ValueError("feature-set training window must be positive")
        if (
            self.evaluation_window is not None
            and self.evaluation_window[0] > self.evaluation_window[1]
        ):
            raise ValueError("feature-set evaluation window is invalid")
        if len(dict(self.lineage)) != len(self.lineage):
            raise ValueError("feature-set lineage keys must be unique")
        if len(self.config_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.config_hash
        ):
            raise ValueError("feature-set config hash must be lowercase SHA-256")
        payload = self.model_dump(exclude={"content_hash", "created_at"}, mode="json")
        expected_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.content_hash and self.content_hash != expected_hash:
            raise ValueError("feature-set content hash does not match its definition")
        object.__setattr__(self, "content_hash", expected_hash)
        return self
