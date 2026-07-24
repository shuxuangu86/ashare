from collections.abc import Callable, Collection
from dataclasses import dataclass
from enum import StrEnum

from aquant.domain.identifiers import Symbol
from aquant.strategies.microcap.models import (
    MicrocapObservation,
    MicrocapSelection,
    MicrocapSnapshot,
    MicrocapUniverseConfig,
)


class MicrocapPrototype(StrEnum):
    SMALLEST_100 = "smallest-100"
    RANK_101_400 = "rank-101-400"
    SMALLEST_400 = "smallest-400"
    EXECUTABLE_95 = "executable-95"
    DIVIDEND_QUALITY_10 = "dividend-quality-10"
    LOW_PB_LOW_TURNOVER_35 = "low-pb-low-turnover-35"


@dataclass(frozen=True, slots=True)
class MicrocapPrototypeDefinition:
    prototype: MicrocapPrototype
    target_count: int
    rank_start: int = 0
    sell_buffer_rank: int | None = None
    executable_capital: bool = False

    def __post_init__(self) -> None:
        if self.target_count <= 0 or self.rank_start < 0:
            raise ValueError("prototype rank and target count are invalid")
        if self.sell_buffer_rank is not None and self.sell_buffer_rank < self.target_count:
            raise ValueError("sell buffer rank cannot be smaller than target count")

    @property
    def minimum_base_constituents(self) -> int:
        return self.rank_start + self.target_count


PROTOTYPE_DEFINITIONS = {
    MicrocapPrototype.SMALLEST_100: MicrocapPrototypeDefinition(
        MicrocapPrototype.SMALLEST_100, 100
    ),
    MicrocapPrototype.RANK_101_400: MicrocapPrototypeDefinition(
        MicrocapPrototype.RANK_101_400, 300, rank_start=100
    ),
    MicrocapPrototype.SMALLEST_400: MicrocapPrototypeDefinition(
        MicrocapPrototype.SMALLEST_400, 400
    ),
    MicrocapPrototype.EXECUTABLE_95: MicrocapPrototypeDefinition(
        MicrocapPrototype.EXECUTABLE_95,
        95,
        sell_buffer_rank=120,
        executable_capital=True,
    ),
    MicrocapPrototype.DIVIDEND_QUALITY_10: MicrocapPrototypeDefinition(
        MicrocapPrototype.DIVIDEND_QUALITY_10, 10, executable_capital=True
    ),
    MicrocapPrototype.LOW_PB_LOW_TURNOVER_35: MicrocapPrototypeDefinition(
        MicrocapPrototype.LOW_PB_LOW_TURNOVER_35, 35, executable_capital=True
    ),
}


class MicrocapPrototypeSelector:
    """Transparent selectors for the six first-round attributable prototypes."""

    def __init__(
        self,
        prototype: MicrocapPrototype,
        *,
        universe: MicrocapUniverseConfig | None = None,
    ) -> None:
        self._definition = PROTOTYPE_DEFINITIONS[prototype]
        self._universe = universe or MicrocapUniverseConfig(
            target_count=400,
            minimum_constituents=400,
        )

    @property
    def definition(self) -> MicrocapPrototypeDefinition:
        return self._definition

    def select(
        self,
        snapshot: MicrocapSnapshot,
        *,
        held_symbols: Collection[Symbol] = (),
    ) -> MicrocapSelection:
        eligible, excluded = self._base_screen(snapshot)
        required = max(
            self._definition.minimum_base_constituents,
            self._universe.minimum_constituents,
        )
        if len(eligible) < required:
            raise ValueError(
                "eligible micro-cap universe is incomplete for "
                f"{self._definition.prototype.value}: {len(eligible)} < {required}"
            )

        ranked = sorted(eligible, key=lambda item: (item.total_market_cap, item.symbol))
        selected, factor_eligible = self._prototype_selection(ranked, held_symbols)
        if len(selected) < self._definition.target_count:
            raise ValueError(
                "factor-eligible micro-cap universe is incomplete for "
                f"{self._definition.prototype.value}: "
                f"{len(selected)} < {self._definition.target_count}"
            )
        excluded["factor_filter"] = len(eligible) - factor_eligible
        return MicrocapSelection(
            snapshot.trade_date,
            snapshot.asof_time,
            tuple(item.symbol for item in selected),
            len(eligible),
            tuple(sorted(excluded.items())),
        )

    def _prototype_selection(
        self,
        ranked: list[MicrocapObservation],
        held_symbols: Collection[Symbol],
    ) -> tuple[list[MicrocapObservation], int]:
        prototype = self._definition.prototype
        if prototype in {
            MicrocapPrototype.SMALLEST_100,
            MicrocapPrototype.RANK_101_400,
            MicrocapPrototype.SMALLEST_400,
        }:
            start = self._definition.rank_start
            stop = start + self._definition.target_count
            return ranked[start:stop], len(ranked)
        if prototype is MicrocapPrototype.EXECUTABLE_95:
            candidates = [
                item
                for item in ranked
                if item.pb is not None
                and item.pb > 0
                and item.net_profit_yoy is not None
                and item.net_profit_yoy > 0
            ]
            return self._buffered_selection(candidates, held_symbols), len(candidates)
        if prototype is MicrocapPrototype.DIVIDEND_QUALITY_10:
            complete = [
                item
                for item in ranked
                if item.dividend_yield is not None
                and item.debt_to_assets is not None
                and item.turnover_volatility_20d is not None
            ]
            high_dividend = self._half(
                complete,
                key=lambda item: self._required(item.dividend_yield),
                reverse=True,
            )
            low_leverage = self._half(
                high_dividend,
                key=lambda item: self._required(item.debt_to_assets),
            )
            low_turnover = self._half(
                low_leverage,
                key=lambda item: self._required(item.turnover_volatility_20d),
            )
            selected = sorted(
                low_turnover,
                key=lambda item: (item.effective_float_market_cap, item.symbol),
            )[: self._definition.target_count]
            return selected, len(low_turnover)
        if prototype is MicrocapPrototype.LOW_PB_LOW_TURNOVER_35:
            complete = [
                item
                for item in ranked
                if item.pb is not None and item.pb > 0 and item.turnover_volatility_20d is not None
            ]
            low_pb = self._half(
                complete,
                key=lambda item: self._required(item.pb),
            )
            low_turnover = self._half(
                low_pb,
                key=lambda item: self._required(item.turnover_volatility_20d),
            )
            selected = sorted(
                low_turnover,
                key=lambda item: (item.total_market_cap, item.symbol),
            )[: self._definition.target_count]
            return selected, len(low_turnover)
        raise AssertionError(f"unhandled micro-cap prototype: {prototype}")

    def _buffered_selection(
        self,
        candidates: list[MicrocapObservation],
        held_symbols: Collection[Symbol],
    ) -> list[MicrocapObservation]:
        buffer_rank = self._definition.sell_buffer_rank
        if buffer_rank is None:
            raise AssertionError("buffered selection requires a sell rank")
        holding_zone = candidates[:buffer_rank]
        held = set(held_symbols)
        retained = [item for item in holding_zone if item.symbol in held]
        retained = retained[: self._definition.target_count]
        retained_symbols = {item.symbol for item in retained}
        additions = [item for item in candidates if item.symbol not in retained_symbols][
            : self._definition.target_count - len(retained)
        ]
        selected = retained + additions
        rank = {item.symbol: index for index, item in enumerate(candidates)}
        return sorted(selected, key=lambda item: (rank[item.symbol], item.symbol))

    @staticmethod
    def _half(
        observations: list[MicrocapObservation],
        *,
        key: Callable[[MicrocapObservation], object],
        reverse: bool = False,
    ) -> list[MicrocapObservation]:
        if not observations:
            return []
        ordered = sorted(
            observations,
            key=lambda item: (key(item), item.symbol),
            reverse=reverse,
        )
        return ordered[: (len(ordered) + 1) // 2]

    @staticmethod
    def _required(value: object | None) -> object:
        if value is None:
            raise AssertionError("factor value was screened before ranking")
        return value

    def _base_screen(
        self, snapshot: MicrocapSnapshot
    ) -> tuple[list[MicrocapObservation], dict[str, int]]:
        eligible: list[MicrocapObservation] = []
        excluded = {
            "delisting_risk": 0,
            "exchange": 0,
            "factor_filter": 0,
            "new_listing": 0,
            "st": 0,
            "suspended": 0,
        }
        for item in snapshot.observations:
            reason = self._exclusion_reason(item, snapshot)
            if reason is None:
                eligible.append(item)
            else:
                excluded[reason] += 1
        return eligible, excluded

    def _exclusion_reason(
        self, item: MicrocapObservation, snapshot: MicrocapSnapshot
    ) -> str | None:
        if item.symbol.exchange not in self._universe.exchanges:
            return "exchange"
        if (snapshot.trade_date - item.list_date).days < self._universe.minimum_listing_days:
            return "new_listing"
        if self._universe.exclude_st and item.is_st:
            return "st"
        if self._universe.exclude_delisting_risk and item.is_delisting_risk:
            return "delisting_risk"
        if self._universe.exclude_suspended and item.suspended:
            return "suspended"
        return None
