from collections.abc import Mapping, Set
from datetime import date
from decimal import ROUND_FLOOR, Decimal

from aquant.backtest.event_engine.engine import StrategyContext
from aquant.backtest.matching import OrderRequest
from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.strategies.microcap.models import (
    MicrocapSelector,
    MicrocapSnapshot,
    MicrocapUniverseConfig,
)
from aquant.strategies.microcap.prototypes import (
    MicrocapPrototype,
    MicrocapPrototypeSelector,
)


class MicrocapEqualWeightStrategy:
    """Research baseline: rank on T, rebalance at the next eligible open."""

    def __init__(
        self,
        *,
        snapshots: Mapping[date, MicrocapSnapshot],
        rebalance_dates: Set[date],
        universe: MicrocapUniverseConfig | None = None,
        lot_size: int = 100,
        cash_buffer_weight: Decimal = Decimal("0.02"),
    ) -> None:
        buffer = Decimal(cash_buffer_weight)
        if lot_size <= 0:
            raise ValueError("lot size must be positive")
        if not Decimal("0") <= buffer < Decimal("1"):
            raise ValueError("cash buffer weight must be in [0, 1)")
        self._snapshots = dict(snapshots)
        self._rebalance_dates = frozenset(rebalance_dates)
        self._selector = MicrocapSelector(universe)
        self._lot_size = lot_size
        self._cash_buffer = buffer

    def on_close(self, context: StrategyContext) -> tuple[OrderRequest, ...]:
        trade_date = context.session.trade_date
        if trade_date not in self._rebalance_dates:
            return ()
        try:
            snapshot = self._snapshots[trade_date]
        except KeyError as exc:
            raise KeyError(f"missing micro-cap PIT snapshot for {trade_date}") from exc
        if snapshot.asof_time > context.session.close_at:
            raise ValueError("micro-cap snapshot was unavailable at signal time")

        selection = self._selector.select(snapshot)
        bars = context.session.bar_by_symbol()
        missing_bars = tuple(symbol for symbol in selection.symbols if symbol not in bars)
        if missing_bars:
            raise ValueError(f"selected micro-cap symbols are missing bars: {missing_bars[:3]}")

        current = {position.symbol: position.quantity for position in context.portfolio.positions}
        investable = context.portfolio.equity * (Decimal("1") - self._cash_buffer)
        per_security = investable / len(selection.symbols)
        target = {
            symbol: self._round_lot(per_security / bars[symbol].close)
            for symbol in selection.symbols
        }
        unbuyable = tuple(symbol for symbol, quantity in target.items() if quantity == 0)
        if unbuyable:
            raise ValueError(
                "capital is insufficient for an executable equal-weight portfolio: "
                f"{len(unbuyable)} targets are below one board lot"
            )
        orders: list[OrderRequest] = []
        for symbol in sorted(set(current) | set(target)):
            delta = target.get(symbol, 0) - current.get(symbol, 0)
            if delta:
                orders.append(
                    OrderRequest(symbol, Side.BUY if delta > 0 else Side.SELL, abs(delta))
                )
        orders.sort(key=lambda order: (order.side is Side.BUY, order.symbol))
        return tuple(orders)

    def _round_lot(self, quantity: Decimal) -> int:
        lots = (quantity / self._lot_size).to_integral_value(rounding=ROUND_FLOOR)
        return int(lots) * self._lot_size


class MicrocapPrototypeStrategy:
    """Equal-weight execution shell for an attributable micro-cap prototype."""

    def __init__(
        self,
        *,
        prototype: MicrocapPrototype,
        snapshots: Mapping[date, MicrocapSnapshot],
        rebalance_dates: Set[date],
        universe: MicrocapUniverseConfig | None = None,
        lot_size: int = 100,
        cash_buffer_weight: Decimal = Decimal("0.02"),
    ) -> None:
        buffer = Decimal(cash_buffer_weight)
        if lot_size <= 0:
            raise ValueError("lot size must be positive")
        if not Decimal("0") <= buffer < Decimal("1"):
            raise ValueError("cash buffer weight must be in [0, 1)")
        self._snapshots = dict(snapshots)
        self._rebalance_dates = frozenset(rebalance_dates)
        self._selector = MicrocapPrototypeSelector(prototype, universe=universe)
        self._lot_size = lot_size
        self._cash_buffer = buffer

    def on_close(self, context: StrategyContext) -> tuple[OrderRequest, ...]:
        trade_date = context.session.trade_date
        current = {position.symbol: position.quantity for position in context.portfolio.positions}
        if trade_date not in self._rebalance_dates:
            return self._daily_hard_exits(context, current)
        try:
            snapshot = self._snapshots[trade_date]
        except KeyError as exc:
            raise KeyError(f"missing micro-cap PIT snapshot for {trade_date}") from exc
        if snapshot.asof_time > context.session.close_at:
            raise ValueError("micro-cap snapshot was unavailable at signal time")

        selection = self._selector.select(snapshot, held_symbols=current)
        bars = context.session.bar_by_symbol()
        missing_bars = tuple(symbol for symbol in selection.symbols if symbol not in bars)
        if missing_bars:
            raise ValueError(f"selected micro-cap symbols are missing bars: {missing_bars[:3]}")

        investable = context.portfolio.equity * (Decimal("1") - self._cash_buffer)
        per_security = investable / len(selection.symbols)
        target = {
            symbol: self._round_lot(per_security / bars[symbol].close)
            for symbol in selection.symbols
        }
        unbuyable = tuple(symbol for symbol, quantity in target.items() if quantity == 0)
        if unbuyable:
            raise ValueError(
                "capital is insufficient for an executable equal-weight portfolio: "
                f"{len(unbuyable)} targets are below one board lot"
            )
        orders = [
            OrderRequest(symbol, Side.BUY if delta > 0 else Side.SELL, abs(delta))
            for symbol in sorted(set(current) | set(target))
            if (delta := target.get(symbol, 0) - current.get(symbol, 0))
        ]
        orders.sort(key=lambda order: (order.side is Side.BUY, order.symbol))
        return tuple(orders)

    def _round_lot(self, quantity: Decimal) -> int:
        lots = (quantity / self._lot_size).to_integral_value(rounding=ROUND_FLOOR)
        return int(lots) * self._lot_size

    def _daily_hard_exits(
        self,
        context: StrategyContext,
        current: dict[Symbol, int],
    ) -> tuple[OrderRequest, ...]:
        if not current:
            return ()
        trade_date = context.session.trade_date
        try:
            snapshot = self._snapshots[trade_date]
        except KeyError as exc:
            raise KeyError(f"missing daily micro-cap risk snapshot for {trade_date}") from exc
        if snapshot.asof_time > context.session.close_at:
            raise ValueError("micro-cap risk snapshot was unavailable at signal time")
        observation_by_symbol = {item.symbol: item for item in snapshot.observations}
        missing = tuple(symbol for symbol in current if symbol not in observation_by_symbol)
        if missing:
            raise ValueError(f"current micro-cap positions are missing risk state: {missing[:3]}")
        exits = (
            OrderRequest(symbol, Side.SELL, quantity)
            for symbol, quantity in current.items()
            if observation_by_symbol[symbol].is_st
            or observation_by_symbol[symbol].is_delisting_risk
        )
        return tuple(sorted(exits, key=lambda order: order.symbol))
