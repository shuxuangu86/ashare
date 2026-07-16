from datetime import datetime
from typing import Protocol, TypeVar

from aquant.data.point_in_time import PointInTimeContext

FactorResultT = TypeVar("FactorResultT", covariant=True)


class Factor(Protocol[FactorResultT]):
    name: str
    version: str
    lookback: int
    required_fields: tuple[str, ...]

    def compute(self, context: PointInTimeContext, asof_time: datetime) -> FactorResultT: ...
