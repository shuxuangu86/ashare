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
    REDUNDANT = "REDUNDANT"
    CONDITIONAL = "CONDITIONAL"
    RISK_ONLY = "RISK_ONLY"
    APPROVED = "APPROVED"
    PAPER_TRADING = "PAPER_TRADING"
    PRODUCTION = "PRODUCTION"
    DECAYED = "DECAYED"
    ARCHIVED = "ARCHIVED"
    DEPRECATED = "DEPRECATED"
    UNAVAILABLE = "UNAVAILABLE"


class SourceType(StrEnum):
    CODE = "CODE"
    FORMULA = "FORMULA"
    GENERATED = "GENERATED"
    PUBLISHED = "PUBLISHED"


class FactorSpec(BaseModel):
    """Immutable, canonical metadata for one version of an L2 factor."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)

    factor_id: str
    name: str
    description: str
    family: str
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
    source_type: SourceType
    source_reference: str
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
        "version",
        "hypothesis",
        "universe",
        "source_reference",
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
