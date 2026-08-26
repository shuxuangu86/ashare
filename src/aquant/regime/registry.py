from __future__ import annotations

from collections.abc import Iterable, Iterator

from aquant.regime.definitions import MarketStateFamily, MarketStateSpec, MarketStateStatus


class MarketStateRegistry:
    """Deterministic registry that removes only exact formula duplicates."""

    def __init__(self, specs: Iterable[MarketStateSpec] = ()) -> None:
        self._specs: dict[tuple[str, str], MarketStateSpec] = {}
        self._formula_owners: dict[str, tuple[str, str]] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: MarketStateSpec) -> None:
        key = (spec.state_id, spec.version)
        if key in self._specs:
            raise ValueError(f"market state {spec.state_id}@{spec.version} already registered")
        formula_key = self._formula_key(spec)
        owner = self._formula_owners.get(formula_key)
        if owner is not None:
            raise ValueError(
                f"exact market-state formula already registered by {owner[0]}@{owner[1]}"
            )
        self._specs[key] = spec
        self._formula_owners[formula_key] = key

    def get(self, state_id: str, version: str = "1.0.0") -> MarketStateSpec:
        try:
            return self._specs[(state_id, version)]
        except KeyError as exc:
            raise KeyError(f"unknown market state {state_id}@{version}") from exc

    def select(
        self,
        *,
        family: MarketStateFamily | None = None,
        status: MarketStateStatus | None = None,
    ) -> tuple[MarketStateSpec, ...]:
        return tuple(
            spec
            for spec in self
            if (family is None or spec.family is family)
            and (status is None or spec.status is status)
        )

    def __iter__(self) -> Iterator[MarketStateSpec]:
        return iter(sorted(self._specs.values(), key=lambda item: (item.state_id, item.version)))

    def __len__(self) -> int:
        return len(self._specs)

    @staticmethod
    def _formula_key(spec: MarketStateSpec) -> str:
        parameters = tuple(sorted((key, repr(value)) for key, value in spec.parameters.items()))
        return repr(
            (
                spec.formula.strip(),
                parameters,
                spec.scope,
                spec.required_indices,
                spec.required_universes,
            )
        )
