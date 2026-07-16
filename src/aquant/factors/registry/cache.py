from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt

from aquant.domain.data_release import DataReleaseId

Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FactorCacheKey:
    expression_hash: str
    asof_date: date
    data_release_id: DataReleaseId
    parameter_hash: str

    def __post_init__(self) -> None:
        for name in ("expression_hash", "parameter_hash"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")


class FactorValueCache:
    """Deterministic cache that refuses conflicting writes for the same lineage key."""

    def __init__(self) -> None:
        self._values: dict[FactorCacheKey, Array] = {}

    def put(self, key: FactorCacheKey, values: Array) -> None:
        resolved = np.asarray(values, dtype=np.float64).copy()
        current = self._values.get(key)
        if current is not None and not np.array_equal(current, resolved, equal_nan=True):
            raise ValueError("factor cache key already contains different values")
        self._values[key] = resolved

    def get(self, key: FactorCacheKey) -> Array | None:
        values = self._values.get(key)
        return None if values is None else values.copy()

    def missing_dates(
        self,
        *,
        expression_hash: str,
        dates: tuple[date, ...],
        data_release_id: DataReleaseId,
        parameter_hash: str,
    ) -> tuple[date, ...]:
        return tuple(
            asof_date
            for asof_date in dates
            if FactorCacheKey(expression_hash, asof_date, data_release_id, parameter_hash)
            not in self._values
        )
