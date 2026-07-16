import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from aquant.domain.time import require_aware


@dataclass(frozen=True, slots=True)
class AuditEvent:
    event_type: str
    payload_json: str
    payload_hash: str
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not self.event_type.strip():
            raise ValueError("event_type must not be blank")
        object.__setattr__(self, "event_type", self.event_type.strip())
        object.__setattr__(
            self, "occurred_at", require_aware(self.occurred_at, field_name="occurred_at")
        )

    @classmethod
    def create(
        cls,
        *,
        event_type: str,
        payload: Mapping[str, Any],
        occurred_at: datetime | None = None,
    ) -> "AuditEvent":
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        kwargs: dict[str, Any] = {
            "event_type": event_type,
            "payload_json": payload_json,
            "payload_hash": payload_hash,
        }
        if occurred_at is not None:
            kwargs["occurred_at"] = occurred_at
        return cls(**kwargs)
