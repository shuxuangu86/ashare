"""Point-in-time market state and regime features."""

from aquant.regime.aggregation import (
    aggregate_turnover,
    amount_share,
    relative_returns,
    returns,
    rolling_mean,
    weighted_mean,
)
from aquant.regime.builder import MarketStateBuilder
from aquant.regime.catalog import core_state_registry
from aquant.regime.crowding import microcap_crowding_score
from aquant.regime.definitions import (
    MarketStateFamily,
    MarketStateReleaseStatus,
    MarketStateRole,
    MarketStateScope,
    MarketStateSpec,
    MarketStateStatus,
    MarketStateValue,
)
from aquant.regime.interactions import RegimeInteractionSpec, build_registered_interactions
from aquant.regime.materialization import materialize_state_values
from aquant.regime.percentiles import expanding_percentile, percentile_rank, rolling_percentile
from aquant.regime.registry import MarketStateRegistry

__all__ = [
    "MarketStateBuilder",
    "MarketStateFamily",
    "MarketStateRegistry",
    "MarketStateReleaseStatus",
    "MarketStateRole",
    "MarketStateScope",
    "MarketStateSpec",
    "MarketStateStatus",
    "MarketStateValue",
    "RegimeInteractionSpec",
    "aggregate_turnover",
    "amount_share",
    "build_registered_interactions",
    "core_state_registry",
    "expanding_percentile",
    "materialize_state_values",
    "microcap_crowding_score",
    "percentile_rank",
    "relative_returns",
    "returns",
    "rolling_mean",
    "rolling_percentile",
    "weighted_mean",
]
