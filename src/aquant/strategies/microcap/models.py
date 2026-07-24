from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from aquant.domain.enums import Exchange
from aquant.domain.identifiers import Symbol
from aquant.domain.time import require_aware


@dataclass(frozen=True, slots=True)
class MicrocapObservation:
    """Point-in-time inputs used by the transparent micro-cap baseline."""

    symbol: Symbol
    trade_date: date
    available_at: datetime
    list_date: date
    total_market_cap: Decimal
    suspended: bool
    is_st: bool

    def __post_init__(self) -> None:
        available_at = require_aware(self.available_at, field_name="available_at")
        market_cap = Decimal(self.total_market_cap)
        if self.list_date > self.trade_date:
            raise ValueError("list date cannot be later than observation date")
        if not market_cap.is_finite() or market_cap <= 0:
            raise ValueError("total market cap must be finite and positive")
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "total_market_cap", market_cap)


@dataclass(frozen=True, slots=True)
class MicrocapSnapshot:
    trade_date: date
    asof_time: datetime
    observations: tuple[MicrocapObservation, ...]

    def __post_init__(self) -> None:
        asof_time = require_aware(self.asof_time, field_name="asof_time")
        if asof_time.date() != self.trade_date:
            raise ValueError("micro-cap snapshot time must fall on its trade date")
        symbols = [item.symbol for item in self.observations]
        if not self.observations or len(symbols) != len(set(symbols)):
            raise ValueError("micro-cap snapshot must contain unique observations")
        if any(item.trade_date != self.trade_date for item in self.observations):
            raise ValueError("micro-cap observation date mismatch")
        if any(item.available_at > asof_time for item in self.observations):
            raise ValueError("micro-cap snapshot contains data unavailable at asof_time")
        object.__setattr__(self, "asof_time", asof_time)


@dataclass(frozen=True, slots=True)
class MicrocapUniverseConfig:
    target_count: int = 400
    minimum_constituents: int = 400
    minimum_listing_days: int = 120
    exclude_st: bool = True
    exclude_suspended: bool = True
    exchanges: frozenset[Exchange] = frozenset({Exchange.XSHG, Exchange.XSHE})

    def __post_init__(self) -> None:
        if self.target_count <= 0:
            raise ValueError("target count must be positive")
        if not 0 < self.minimum_constituents <= self.target_count:
            raise ValueError("minimum constituents must be within target count")
        if self.minimum_listing_days < 0:
            raise ValueError("minimum listing days cannot be negative")
        if not self.exchanges:
            raise ValueError("at least one exchange must be eligible")


@dataclass(frozen=True, slots=True)
class MicrocapSelection:
    trade_date: date
    asof_time: datetime
    symbols: tuple[Symbol, ...]
    eligible_count: int
    excluded_counts: tuple[tuple[str, int], ...]


class MicrocapSelector:
    """Deterministic smallest-market-cap selector with explicit exclusion counts."""

    def __init__(self, config: MicrocapUniverseConfig | None = None) -> None:
        self._config = config or MicrocapUniverseConfig()

    def select(self, snapshot: MicrocapSnapshot) -> MicrocapSelection:
        eligible: list[MicrocapObservation] = []
        excluded = {
            "exchange": 0,
            "new_listing": 0,
            "st": 0,
            "suspended": 0,
        }
        for item in snapshot.observations:
            reason = self._exclusion_reason(item, snapshot.trade_date)
            if reason is None:
                eligible.append(item)
            else:
                excluded[reason] += 1
        if len(eligible) < self._config.minimum_constituents:
            raise ValueError(
                "eligible micro-cap universe is incomplete: "
                f"{len(eligible)} < {self._config.minimum_constituents}"
            )
        ranked = sorted(eligible, key=lambda item: (item.total_market_cap, item.symbol))
        return MicrocapSelection(
            snapshot.trade_date,
            snapshot.asof_time,
            tuple(item.symbol for item in ranked[: self._config.target_count]),
            len(eligible),
            tuple(sorted(excluded.items())),
        )

    def _exclusion_reason(self, item: MicrocapObservation, trade_date: date) -> str | None:
        if item.symbol.exchange not in self._config.exchanges:
            return "exchange"
        if (trade_date - item.list_date).days < self._config.minimum_listing_days:
            return "new_listing"
        if self._config.exclude_st and item.is_st:
            return "st"
        if self._config.exclude_suspended and item.suspended:
            return "suspended"
        return None
