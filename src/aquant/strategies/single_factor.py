import math
from collections.abc import Mapping, Set
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_FLOOR, Decimal
from enum import StrEnum

from aquant.backtest.event_engine.engine import StrategyContext
from aquant.backtest.matching import OrderRequest
from aquant.domain.enums import Exchange, Side
from aquant.domain.identifiers import Symbol
from aquant.domain.time import require_aware


class SelectionTail(StrEnum):
    BEST = "best"
    WORST = "worst"


@dataclass(frozen=True, slots=True)
class SingleFactorObservation:
    symbol: Symbol
    trade_date: date
    available_at: datetime
    list_date: date
    factor_value: float | None
    suspended: bool
    is_st: bool
    is_delisting_risk: bool

    def __post_init__(self) -> None:
        available_at = require_aware(self.available_at, field_name="available_at")
        if available_at.date() != self.trade_date or self.list_date > self.trade_date:
            raise ValueError("single-factor observation dates are invalid")
        if self.factor_value is not None and not math.isfinite(self.factor_value):
            raise ValueError("single-factor value must be finite when populated")
        object.__setattr__(self, "available_at", available_at)


@dataclass(frozen=True, slots=True)
class SingleFactorSnapshot:
    trade_date: date
    asof_time: datetime
    observations: tuple[SingleFactorObservation, ...]

    def __post_init__(self) -> None:
        asof_time = require_aware(self.asof_time, field_name="asof_time")
        symbols = [item.symbol for item in self.observations]
        if asof_time.date() != self.trade_date:
            raise ValueError("single-factor snapshot must be timestamped on trade_date")
        if not self.observations or len(symbols) != len(set(symbols)):
            raise ValueError("single-factor snapshot observations must be non-empty and unique")
        if any(item.trade_date != self.trade_date for item in self.observations):
            raise ValueError("single-factor observation date mismatch")
        if any(item.available_at > asof_time for item in self.observations):
            raise ValueError("single-factor snapshot contains unavailable data")
        object.__setattr__(self, "asof_time", asof_time)


@dataclass(frozen=True, slots=True)
class SingleFactorConfig:
    factor_id: str
    factor_version: str
    expected_direction: int
    target_count: int = 50
    minimum_constituents: int = 30
    minimum_listing_days: int = 120
    lot_size: int = 100
    cash_buffer_weight: Decimal = Decimal("0.02")
    target_fraction: Decimal | None = None
    selection_tail: SelectionTail = SelectionTail.BEST

    def __post_init__(self) -> None:
        factor_id = self.factor_id.strip()
        factor_version = self.factor_version.strip()
        cash_buffer = Decimal(self.cash_buffer_weight)
        target_fraction = (
            Decimal(self.target_fraction) if self.target_fraction is not None else None
        )
        if not factor_id or not factor_version:
            raise ValueError("single-factor identity must not be blank")
        if self.expected_direction not in {-1, 1}:
            raise ValueError("single-factor expected direction must be -1 or 1")
        if self.target_count <= 0 or self.minimum_constituents <= 0:
            raise ValueError("single-factor constituent counts are invalid")
        if target_fraction is not None and not Decimal("0") < target_fraction < Decimal("1"):
            raise ValueError("single-factor target fraction must be in (0, 1)")
        if self.minimum_listing_days < 0 or self.lot_size <= 0:
            raise ValueError("single-factor listing days and lot size are invalid")
        if not Decimal("0") <= cash_buffer < Decimal("1"):
            raise ValueError("single-factor cash buffer must be in [0, 1)")
        object.__setattr__(self, "factor_id", factor_id)
        object.__setattr__(self, "factor_version", factor_version)
        object.__setattr__(self, "cash_buffer_weight", cash_buffer)
        object.__setattr__(self, "target_fraction", target_fraction)
        object.__setattr__(self, "selection_tail", SelectionTail(self.selection_tail))


@dataclass(frozen=True, slots=True)
class SingleFactorSelection:
    trade_date: date
    symbols: tuple[Symbol, ...]
    eligible_count: int
    excluded_counts: tuple[tuple[str, int], ...]


class SingleFactorSelector:
    def __init__(self, config: SingleFactorConfig) -> None:
        self.config = config

    def select(self, snapshot: SingleFactorSnapshot) -> SingleFactorSelection:
        eligible: list[SingleFactorObservation] = []
        excluded = {
            "delisting_risk": 0,
            "exchange": 0,
            "missing_factor": 0,
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
        if len(eligible) < self.config.minimum_constituents:
            raise ValueError(
                "eligible single-factor universe is incomplete: "
                f"{len(eligible)} < {self.config.minimum_constituents}"
            )
        direction = self.config.expected_direction
        if self.config.selection_tail is SelectionTail.WORST:
            direction = -direction
        ranked = sorted(
            eligible,
            key=lambda item: self._ranking_key(item, direction),
        )
        target_count = self.config.target_count
        if self.config.target_fraction is not None:
            target_count = max(1, int(len(eligible) * self.config.target_fraction))
        return SingleFactorSelection(
            snapshot.trade_date,
            tuple(item.symbol for item in ranked[:target_count]),
            len(eligible),
            tuple(sorted(excluded.items())),
        )

    @staticmethod
    def _ranking_key(
        item: SingleFactorObservation,
        direction: int,
    ) -> tuple[float, Symbol]:
        assert item.factor_value is not None
        return -direction * item.factor_value, item.symbol

    def _exclusion_reason(
        self,
        item: SingleFactorObservation,
        trade_date: date,
    ) -> str | None:
        if item.symbol.exchange not in {Exchange.XSHG, Exchange.XSHE}:
            return "exchange"
        if item.factor_value is None:
            return "missing_factor"
        if item.suspended:
            return "suspended"
        if item.is_st:
            return "st"
        if item.is_delisting_risk:
            return "delisting_risk"
        if (trade_date - item.list_date).days < self.config.minimum_listing_days:
            return "new_listing"
        return None


class SingleFactorEqualWeightStrategy:
    """Rank at T close and submit deterministic equal-weight orders for T+1 open."""

    def __init__(
        self,
        *,
        snapshots: Mapping[date, SingleFactorSnapshot],
        rebalance_dates: Set[date],
        config: SingleFactorConfig,
    ) -> None:
        self._snapshots = dict(snapshots)
        self._rebalance_dates = frozenset(rebalance_dates)
        self._config = config
        self._selector = SingleFactorSelector(config)

    def on_close(self, context: StrategyContext) -> tuple[OrderRequest, ...]:
        trade_date = context.session.trade_date
        snapshot = self._snapshot(trade_date, context.session.close_at)
        current = {position.symbol: position.quantity for position in context.portfolio.positions}
        if trade_date not in self._rebalance_dates:
            return self._hard_exits(snapshot, current)

        selection = self._selector.select(snapshot)
        bars = context.session.bar_by_symbol()
        missing_bars = tuple(symbol for symbol in selection.symbols if symbol not in bars)
        if missing_bars:
            raise ValueError(f"selected single-factor symbols are missing bars: {missing_bars[:3]}")
        investable = context.portfolio.equity * (Decimal("1") - self._config.cash_buffer_weight)
        per_security = investable / len(selection.symbols)
        target = {
            symbol: self._round_lot(per_security / bars[symbol].close)
            for symbol in selection.symbols
        }
        if any(quantity <= 0 for quantity in target.values()):
            raise ValueError("single-factor capital is insufficient for one board lot per target")
        orders = [
            OrderRequest(symbol, Side.BUY if delta > 0 else Side.SELL, abs(delta))
            for symbol in sorted(set(current) | set(target))
            if (delta := target.get(symbol, 0) - current.get(symbol, 0))
        ]
        orders.sort(key=lambda order: (order.side is Side.BUY, order.symbol))
        return tuple(orders)

    def _snapshot(self, trade_date: date, close_at: datetime) -> SingleFactorSnapshot:
        try:
            snapshot = self._snapshots[trade_date]
        except KeyError as exc:
            raise KeyError(f"missing single-factor PIT snapshot for {trade_date}") from exc
        if snapshot.asof_time > close_at:
            raise ValueError("single-factor snapshot was unavailable at signal time")
        return snapshot

    def _round_lot(self, quantity: Decimal) -> int:
        lots = (quantity / self._config.lot_size).to_integral_value(rounding=ROUND_FLOOR)
        return int(lots) * self._config.lot_size

    @staticmethod
    def _hard_exits(
        snapshot: SingleFactorSnapshot,
        current: dict[Symbol, int],
    ) -> tuple[OrderRequest, ...]:
        if not current:
            return ()
        by_symbol = {item.symbol: item for item in snapshot.observations}
        missing = tuple(symbol for symbol in current if symbol not in by_symbol)
        if missing:
            raise ValueError(f"single-factor positions are missing risk state: {missing[:3]}")
        exits = (
            OrderRequest(symbol, Side.SELL, quantity)
            for symbol, quantity in current.items()
            if by_symbol[symbol].is_st or by_symbol[symbol].is_delisting_risk
        )
        return tuple(sorted(exits, key=lambda order: order.symbol))
