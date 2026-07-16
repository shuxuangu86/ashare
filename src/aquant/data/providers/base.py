import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from aquant.domain.time import require_aware


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class DatasetRequest:
    dataset: str
    params_json: str
    requested_at: datetime

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise ValueError("dataset must not be blank")
        parsed = json.loads(self.params_json)
        if not isinstance(parsed, dict):
            raise ValueError("dataset request parameters must be a JSON object")
        object.__setattr__(self, "dataset", self.dataset.strip().lower())
        object.__setattr__(self, "params_json", _canonical_json(parsed))
        object.__setattr__(
            self,
            "requested_at",
            require_aware(self.requested_at, field_name="requested_at"),
        )

    @classmethod
    def create(
        cls, *, dataset: str, params: dict[str, Any], requested_at: datetime
    ) -> "DatasetRequest":
        return cls(
            dataset=dataset,
            params_json=_canonical_json(params),
            requested_at=requested_at,
        )

    @property
    def params(self) -> dict[str, Any]:
        value = json.loads(self.params_json)
        if not isinstance(value, dict):  # pragma: no cover - guarded by construction
            raise AssertionError("canonical request parameters must be an object")
        return value


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    provider: str
    provider_version: str
    request: DatasetRequest
    received_at: datetime
    body: bytes
    media_type: str
    source_request_id: str
    record_count: int | None = None
    transport_status: int | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.provider_version.strip():
            raise ValueError("provider and provider_version must not be blank")
        if not self.source_request_id.strip():
            raise ValueError("source_request_id must not be blank")
        if self.record_count is not None and self.record_count < 0:
            raise ValueError("record_count cannot be negative")
        if self.transport_status is not None and not 100 <= self.transport_status <= 599:
            raise ValueError("transport_status must be a valid HTTP status")
        received_at = require_aware(self.received_at, field_name="received_at")
        if received_at < self.request.requested_at:
            raise ValueError("received_at cannot be earlier than requested_at")
        object.__setattr__(self, "provider", self.provider.strip().lower())
        object.__setattr__(self, "provider_version", self.provider_version.strip())
        object.__setattr__(self, "media_type", self.media_type.strip().lower())
        object.__setattr__(self, "source_request_id", self.source_request_id.strip())
        object.__setattr__(self, "received_at", received_at)


@runtime_checkable
class MarketDataProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def fetch(self, request: DatasetRequest) -> ProviderResponse: ...
