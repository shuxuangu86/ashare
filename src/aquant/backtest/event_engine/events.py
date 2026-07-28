from dataclasses import dataclass, field
from datetime import date, datetime
from enum import IntEnum, StrEnum
from typing import Protocol

from aquant.domain.corporate_actions import CorporateAction
from aquant.domain.time import require_aware


class EventKind(StrEnum):
    SESSION_OPEN = "SESSION_OPEN"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    FILL = "FILL"
    SESSION_CLOSE = "SESSION_CLOSE"
    CORPORATE_ACTION = "CORPORATE_ACTION"


class EventPriority(IntEnum):
    SESSION_OPEN = 10
    FILL = 20
    ORDER = 30
    SESSION_CLOSE = 40


class BacktestEvent(Protocol):
    @property
    def kind(self) -> EventKind: ...

    @property
    def occurred_at(self) -> datetime: ...


@dataclass(frozen=True, slots=True)
class SessionOpenEvent:
    trade_date: date
    occurred_at: datetime
    kind: EventKind = field(default=EventKind.SESSION_OPEN, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "occurred_at",
            require_aware(self.occurred_at, field_name="occurred_at"),
        )


@dataclass(frozen=True, slots=True)
class SessionCloseEvent:
    trade_date: date
    occurred_at: datetime
    kind: EventKind = field(default=EventKind.SESSION_CLOSE, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "occurred_at",
            require_aware(self.occurred_at, field_name="occurred_at"),
        )


@dataclass(frozen=True, slots=True)
class CorporateActionAppliedEvent:
    action: CorporateAction
    kind: EventKind = field(default=EventKind.CORPORATE_ACTION, init=False)

    @property
    def occurred_at(self) -> datetime:
        return self.action.occurred_at
