from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aquant.factors.definitions.dsl import FactorExpressionError, parse_expression


class FactorLayer(StrEnum):
    L2A = "L2A"
    L2B = "L2B"
    L2C = "L2C"
    L2R = "L2R"


class FactorStatus(StrEnum):
    CANDIDATE = "CANDIDATE"
    DRAFT = "DRAFT"
    COMPUTED = "COMPUTED"
    VALIDATED = "VALIDATED"
    RESEARCH_VALIDATED = "RESEARCH_VALIDATED"
    FEATURE_ELIGIBLE = "FEATURE_ELIGIBLE"
    STANDALONE_PRODUCTION_ALPHA = "STANDALONE_PRODUCTION_ALPHA"
    REDUNDANT = "REDUNDANT"
    CONDITIONAL = "CONDITIONAL"
    RISK_ONLY = "RISK_ONLY"
    APPROVED = "APPROVED"
    PAPER_TRADING = "PAPER_TRADING"
    PRODUCTION = "PRODUCTION"
    DECAYED = "DECAYED"
    ARCHIVED = "ARCHIVED"
    DEPRECATED = "DEPRECATED"
    REJECTED = "REJECTED"
    UNAVAILABLE = "UNAVAILABLE"


class SourceType(StrEnum):
    CODE = "CODE"
    FORMULA = "FORMULA"
    GENERATED = "GENERATED"
    PUBLISHED = "PUBLISHED"
    ACADEMIC_PAPER = "ACADEMIC_PAPER"
    BROKER_REPORT = "BROKER_REPORT"
    FORMULA_LIBRARY = "FORMULA_LIBRARY"
    TECHNICAL_INDICATOR_STANDARD = "TECHNICAL_INDICATOR_STANDARD"
    OPEN_SOURCE_REFERENCE = "OPEN_SOURCE_REFERENCE"


class FactorRole(StrEnum):
    ALPHA_CANDIDATE = "ALPHA_CANDIDATE"
    RISK_FACTOR = "RISK_FACTOR"
    CONTROL_FEATURE = "CONTROL_FEATURE"
    STATE_FEATURE = "STATE_FEATURE"
    UNKNOWN = "UNKNOWN"


class SourceFaithfulness(StrEnum):
    EXACT = "EXACT"
    NORMALIZED_EQUIVALENT = "NORMALIZED_EQUIVALENT"
    A_SHARE_ADAPTED = "A_SHARE_ADAPTED"
    CORRECTED_AMBIGUITY = "CORRECTED_AMBIGUITY"
    DERIVED_VARIANT = "DERIVED_VARIANT"


class ImplementationStatus(StrEnum):
    IMPLEMENTED = "IMPLEMENTED"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    FORMULA_AMBIGUOUS = "FORMULA_AMBIGUOUS"
    DATA_DEPENDENCY_MISSING = "DATA_DEPENDENCY_MISSING"
    IMPLEMENTATION_FAILED = "IMPLEMENTATION_FAILED"
    DEFERRED_INTRADAY = "DEFERRED_INTRADAY"


class FactorSpec(BaseModel):
    """Immutable, canonical metadata for one version of an L2 factor."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)

    factor_id: str
    name: str
    description: str
    family: str
    subfamily: str = "unspecified"
    role: FactorRole = FactorRole.UNKNOWN
    layer: FactorLayer
    version: str
    status: FactorStatus = FactorStatus.DRAFT
    hypothesis: str
    expected_direction: int = Field(ge=-1, le=1)
    expression: str | None = None
    implementation: str | None = None
    input_fields: tuple[str, ...]
    parent_factor_ids: tuple[str, ...] = ()
    variant_dimension: str | None = None
    required_history: int = Field(ge=0)
    data_lag: int = Field(ge=0)
    universe: str
    valid_from: date | None = None
    valid_to: date | None = None
    target_horizons: tuple[int, ...] = ()
    preprocessing: tuple[str, ...] = ()
    neutralization: tuple[str, ...] = ()
    parameters: dict[str, Any] = Field(default_factory=dict)
    required_datasets: tuple[str, ...] = ("bars_1d",)
    minimum_periods: int | None = Field(default=None, ge=1)
    availability_lag: int | None = Field(default=None, ge=0)
    normalization: str = "NONE"
    missing_policy: str = "PRESERVE"
    warmup_policy: str = "REQUIRE_MINIMUM_PERIODS"
    source_type: SourceType
    source_reference: str
    source_id: str = "SRC_AQUANT_INTERNAL"
    source_section: str | None = None
    source_formula_id: str | None = None
    source_page: str | None = None
    source_faithfulness: SourceFaithfulness = SourceFaithfulness.DERIVED_VARIANT
    implementation_notes: str = ""
    variant_of: str | None = None
    implementation_status: ImplementationStatus = ImplementationStatus.IMPLEMENTED
    tags: tuple[str, ...] = ()
    complexity_score: float = Field(ge=0)
    expression_hash: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator(
        "factor_id",
        "name",
        "description",
        "family",
        "subfamily",
        "version",
        "hypothesis",
        "universe",
        "source_reference",
        "source_id",
        "normalization",
        "missing_policy",
        "warmup_policy",
    )
    @classmethod
    def _not_blank(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("factor metadata must not be blank")
        return resolved

    @field_validator("input_fields", "parent_factor_ids", "target_horizons", "tags")
    @classmethod
    def _unique_tuple(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(value) != len(set(value)):
            raise ValueError("factor metadata lists must not contain duplicates")
        return value

    @field_validator("required_datasets")
    @classmethod
    def _unique_datasets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError("required_datasets must be non-empty and unique")
        return value

    @field_validator("target_horizons")
    @classmethod
    def _positive_horizons(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(item <= 0 for item in value):
            raise ValueError("target horizons must be positive")
        return tuple(sorted(value))

    @field_validator("created_at", "updated_at")
    @classmethod
    def _aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("factor timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_definition(self) -> FactorSpec:
        if bool(self.expression) == bool(self.implementation):
            raise ValueError("exactly one of expression or implementation is required")
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("factor validity interval is inverted")
        if self.factor_id in self.parent_factor_ids:
            raise ValueError("factor cannot be its own parent")
        if self.minimum_periods is None:
            object.__setattr__(self, "minimum_periods", max(1, self.required_history))
        if self.availability_lag is None:
            object.__setattr__(self, "availability_lag", self.data_lag)
        canonical = self.canonical_expression
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        if self.expression_hash and self.expression_hash != digest:
            raise ValueError("expression_hash does not match canonical definition")
        object.__setattr__(self, "expression_hash", digest)
        return self

    @property
    def canonical_expression(self) -> str:
        if self.expression is not None:
            try:
                return parse_expression(self.expression.strip()).canonical
            except FactorExpressionError as exc:
                raise ValueError(str(exc)) from exc
        payload = {
            "implementation": self.implementation,
            "parameters": self.parameters,
            "input_fields": sorted(self.input_fields),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def to_yaml(self) -> str:
        payload = json.loads(self.model_dump_json())
        return yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)

    @classmethod
    def from_json(cls, value: str) -> FactorSpec:
        return cls.model_validate_json(value)

    @classmethod
    def from_yaml(cls, value: str) -> FactorSpec:
        payload = yaml.safe_load(value)
        if not isinstance(payload, dict):
            raise ValueError("factor YAML must contain a mapping")
        return cls.model_validate(payload)

    def write_json(self, path: Path) -> None:
        path.write_text(self.to_json() + "\n", encoding="utf-8")

    def write_yaml(self, path: Path) -> None:
        path.write_text(self.to_yaml(), encoding="utf-8")
