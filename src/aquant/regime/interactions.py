from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from aquant.regime.protocol import Numeric, StateSeries


@dataclass(frozen=True, slots=True)
class RegimeInteractionSpec:
    stock_factor_family: str
    market_state_family: str
    interaction_id: str

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.stock_factor_family,
                self.market_state_family,
                self.interaction_id,
            )
        ):
            raise ValueError("interaction metadata must not be blank")


def build_registered_interactions(
    stock_factors: Mapping[str, Sequence[Numeric]],
    market_states: Mapping[str, Sequence[Numeric]],
    registrations: Sequence[RegimeInteractionSpec],
    *,
    maximum_interactions: int = 32,
) -> Mapping[str, StateSeries]:
    if len(registrations) > maximum_interactions:
        raise ValueError("registered interactions exceed complexity limit")
    output: dict[str, StateSeries] = {}
    for registration in registrations:
        if registration.interaction_id in output:
            raise ValueError("interaction ids must be unique")
        try:
            stock = stock_factors[registration.stock_factor_family]
            state = market_states[registration.market_state_family]
        except KeyError as exc:
            raise KeyError(f"missing registered interaction input: {exc.args[0]}") from exc
        if len(stock) != len(state):
            raise ValueError("interaction inputs must have equal length")
        output[registration.interaction_id] = tuple(
            None
            if stock_value is None or state_value is None
            else float(stock_value) * float(state_value)
            for stock_value, state_value in zip(stock, state, strict=True)
        )
    return output
