from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Protocol

from aquant.regime.definitions import MarketStateSpec

Numeric = float | int | None
StateSeries = tuple[float | None, ...]


class MarketStateCalculator(Protocol):
    spec: MarketStateSpec

    def compute(
        self,
        dates: Sequence[date],
        inputs: Mapping[str, Sequence[Numeric]],
    ) -> StateSeries:
        """Compute in ascending date order using only values visible at each date."""
        ...
