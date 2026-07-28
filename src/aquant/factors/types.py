from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol

from aquant.domain.data_release import DataReleaseId
from aquant.domain.time import require_aware


class FactorInputLoader(Protocol):
    def load(
        self,
        *,
        fields: tuple[str, ...],
        start_date: date,
        end_date: date,
        as_of_time: datetime,
        universe_id: str,
        data_release_id: DataReleaseId,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class FactorContext:
    start_date: date
    end_date: date
    as_of_time: datetime
    universe_id: str
    data_release_id: DataReleaseId
    input_loader: FactorInputLoader
    calendar: Any
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("factor context date range is inverted")
        if not self.universe_id.strip():
            raise ValueError("factor context universe must not be blank")
        object.__setattr__(
            self,
            "as_of_time",
            require_aware(self.as_of_time, field_name="as_of_time"),
        )


@dataclass(frozen=True, slots=True)
class FactorValue:
    trade_date: date
    ts_code: str
    factor_id: str
    factor_version: str
    value: float | None
    is_valid: bool
    quality_flags: tuple[str, ...]
    data_release_id: DataReleaseId
    computed_at: datetime

    def __post_init__(self) -> None:
        if any(not item.strip() for item in (self.ts_code, self.factor_id, self.factor_version)):
            raise ValueError("factor result identifiers must not be blank")
        object.__setattr__(
            self,
            "computed_at",
            require_aware(self.computed_at, field_name="computed_at"),
        )


@dataclass(frozen=True, slots=True)
class FactorResult:
    values: tuple[FactorValue, ...]

    def __post_init__(self) -> None:
        keys = [(v.trade_date, v.ts_code, v.factor_id, v.factor_version) for v in self.values]
        if len(keys) != len(set(keys)):
            raise ValueError("factor result primary key must be unique")
        if keys != sorted(keys):
            raise ValueError("factor result must have deterministic primary-key ordering")
