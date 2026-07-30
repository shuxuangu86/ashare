from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MarketStateFamily(StrEnum):
    STYLE = "STYLE"
    INDEX_RELATIVE = "INDEX_RELATIVE"
    LIQUIDITY = "LIQUIDITY"
    VALUATION = "VALUATION"
    CROWDING = "CROWDING"
    TREND = "TREND"
    BREADTH = "BREADTH"
    VOLATILITY = "VOLATILITY"
    RISK_APPETITE = "RISK_APPETITE"
    FUNDAMENTALS = "FUNDAMENTALS"


class MarketStateScope(StrEnum):
    ALL_A = "ALL_A"
    HS300 = "HS300"
    CSI500 = "CSI500"
    CSI800 = "CSI800"
    CSI1000 = "CSI1000"
    CSI2000 = "CSI2000"
    MICROCAP = "MICROCAP"
    GROWTH = "GROWTH"
    VALUE = "VALUE"
    DIVIDEND = "DIVIDEND"
    TECHNOLOGY = "TECHNOLOGY"
    CONSUMPTION = "CONSUMPTION"
    FINANCIAL = "FINANCIAL"
    CUSTOM_DYNAMIC_UNIVERSE = "CUSTOM_DYNAMIC_UNIVERSE"


class MarketStateRole(StrEnum):
    STATE_FEATURE = "STATE_FEATURE"
    REGIME_INPUT = "REGIME_INPUT"
    RISK_STATE = "RISK_STATE"
    STYLE_STATE = "STYLE_STATE"
    LIQUIDITY_STATE = "LIQUIDITY_STATE"
    CROWDING_STATE = "CROWDING_STATE"
    VALUATION_STATE = "VALUATION_STATE"


class MarketStateStatus(StrEnum):
    IMPLEMENTED = "IMPLEMENTED"
    PARTIAL = "PARTIAL"
    DATA_DEPENDENCY_MISSING = "DATA_DEPENDENCY_MISSING"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    FORMULA_AMBIGUOUS = "FORMULA_AMBIGUOUS"
    DEFERRED = "DEFERRED"
    FAILED = "FAILED"


class MarketStateReleaseStatus(StrEnum):
    DRAFT = "DRAFT"
    RESEARCH_VALIDATED = "RESEARCH_VALIDATED"


class MarketStateSpec(BaseModel):
    """Immutable definition of a point-in-time market state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state_id: str
    version: str = "1.0.0"
    display_name: str
    family: MarketStateFamily
    subfamily: str
    scope: MarketStateScope
    formula: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    required_datasets: tuple[str, ...]
    required_indices: tuple[str, ...] = ()
    required_universes: tuple[str, ...] = ()
    lookback: int = Field(ge=0)
    minimum_periods: int = Field(ge=1)
    availability_lag: int = Field(default=1, ge=1)
    aggregation_method: str
    normalization: str = "NONE"
    percentile_method: str = "NONE"
    expected_interpretation: str
    economic_rationale: str
    source: str = "AQuant Market State Layer v1"
    role: MarketStateRole = MarketStateRole.STATE_FEATURE
    missing_policy: str = "PRESERVE"
    warmup_policy: str = "REQUIRE_MINIMUM_PERIODS"
    status: MarketStateStatus = MarketStateStatus.IMPLEMENTED
    release_status: MarketStateReleaseStatus = MarketStateReleaseStatus.DRAFT
    correlation_cluster: str | None = None
    related_states: tuple[str, ...] = ()
    content_hash: str = ""

    @field_validator(
        "state_id",
        "version",
        "display_name",
        "subfamily",
        "formula",
        "aggregation_method",
        "normalization",
        "percentile_method",
        "expected_interpretation",
        "economic_rationale",
        "source",
        "missing_policy",
        "warmup_policy",
    )
    @classmethod
    def _not_blank(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("market-state metadata must not be blank")
        return resolved

    @field_validator(
        "required_datasets",
        "required_indices",
        "required_universes",
        "related_states",
    )
    @classmethod
    def _unique_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError("market-state metadata lists must be unique and non-blank")
        return value

    @model_validator(mode="after")
    def _validate_definition(self) -> MarketStateSpec:
        if not self.required_datasets:
            raise ValueError("market state must declare at least one required dataset")
        if self.minimum_periods > max(1, self.lookback):
            raise ValueError("minimum_periods cannot exceed lookback")
        if self.state_id in self.related_states:
            raise ValueError("market state cannot be related to itself")
        payload = self.model_dump(exclude={"content_hash"}, mode="json")
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if self.content_hash and self.content_hash != expected:
            raise ValueError("content_hash does not match market-state definition")
        object.__setattr__(self, "content_hash", expected)
        return self


class MarketStateValue(BaseModel):
    """A materialized daily value that is unavailable for same-session trading."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trade_date: date
    available_date: date
    state_id: str
    state_version: str
    scope: MarketStateScope
    raw_value: float | None
    normalized_value: float | None = None
    percentile_value: float | None = None
    status: MarketStateStatus
    data_release_id: str
    code_version: str
    config_hash: str
    content_hash: str = ""

    @field_validator("state_id", "state_version", "data_release_id", "code_version")
    @classmethod
    def _value_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("market-state value identity must not be blank")
        return value

    @field_validator("config_hash")
    @classmethod
    def _sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("config_hash must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _validate_value(self) -> MarketStateValue:
        if self.available_date <= self.trade_date:
            raise ValueError("close-derived market state must not be available on trade_date")
        if self.percentile_value is not None and not 0.0 <= self.percentile_value <= 1.0:
            raise ValueError("percentile_value must be in [0, 1]")
        payload = self.model_dump(exclude={"content_hash"}, mode="json")
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if self.content_hash and self.content_hash != expected:
            raise ValueError("content_hash does not match market-state value")
        object.__setattr__(self, "content_hash", expected)
        return self
