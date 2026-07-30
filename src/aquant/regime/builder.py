from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from itertools import pairwise

from aquant.regime.definitions import MarketStateSpec
from aquant.regime.protocol import Numeric, StateSeries
from aquant.regime.registry import MarketStateRegistry


class MarketStateBuilder:
    """Runs registered calculators after enforcing aligned, ascending PIT inputs."""

    def __init__(self, registry: MarketStateRegistry) -> None:
        self.registry = registry

    @staticmethod
    def validate_inputs(
        dates: Sequence[date],
        inputs: Mapping[str, Sequence[Numeric]],
    ) -> None:
        if any(left >= right for left, right in pairwise(dates)):
            raise ValueError("market-state dates must be strictly increasing")
        mismatched = sorted(name for name, values in inputs.items() if len(values) != len(dates))
        if mismatched:
            raise ValueError(f"market-state input length mismatch: {mismatched}")

    def select_specs(self, state_ids: Iterable[str] | None = None) -> tuple[MarketStateSpec, ...]:
        if state_ids is None:
            return tuple(self.registry)
        requested = tuple(state_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("requested state ids must be unique")
        return tuple(self.registry.get(state_id) for state_id in requested)

    @staticmethod
    def require_output_length(
        dates: Sequence[date],
        outputs: Mapping[str, StateSeries],
    ) -> None:
        mismatched = sorted(name for name, values in outputs.items() if len(values) != len(dates))
        if mismatched:
            raise ValueError(f"market-state output length mismatch: {mismatched}")
