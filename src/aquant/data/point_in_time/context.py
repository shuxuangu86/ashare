from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from aquant.data.point_in_time.table import PointInTimeRecord, PointInTimeTable
from aquant.domain.data_release import DataReleaseId
from aquant.domain.time import require_aware


@dataclass(frozen=True, slots=True)
class PointInTimeContext:
    asof_time: datetime
    data_release_id: DataReleaseId
    tables: Mapping[str, PointInTimeTable[Any, Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof_time", require_aware(self.asof_time, field_name="asof_time"))

    def latest(self, dataset: str, key: Hashable) -> PointInTimeRecord[Any, Any] | None:
        try:
            table = self.tables[dataset]
        except KeyError as exc:
            raise KeyError(f"dataset is not registered in PIT context: {dataset}") from exc
        return table.latest(key, self.asof_time)

    def snapshot(self, dataset: str) -> dict[Hashable, PointInTimeRecord[Any, Any]]:
        try:
            table = self.tables[dataset]
        except KeyError as exc:
            raise KeyError(f"dataset is not registered in PIT context: {dataset}") from exc
        return table.latest_by_key(self.asof_time)
