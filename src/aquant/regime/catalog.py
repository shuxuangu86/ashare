from aquant.regime.crowding import crowding_state_specs
from aquant.regime.extended_catalog import extended_state_specs
from aquant.regime.index_relative import index_relative_state_specs
from aquant.regime.liquidity import liquidity_state_specs
from aquant.regime.registry import MarketStateRegistry
from aquant.regime.style import style_state_specs
from aquant.regime.valuation import valuation_state_specs


def core_state_registry() -> MarketStateRegistry:
    return MarketStateRegistry(
        (
            *style_state_specs(),
            *index_relative_state_specs(),
            *liquidity_state_specs(),
            *valuation_state_specs(),
            *crowding_state_specs(),
            *extended_state_specs(),
        )
    )
