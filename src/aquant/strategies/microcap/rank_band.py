from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_FLOOR, Decimal

from aquant.backtest import OrderRequest, StrategyContext
from aquant.domain.corporate_actions import CorporateActionKind
from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol


@dataclass(frozen=True, slots=True)
class MicrocapRankBandTarget:
    """One ordered PIT candidate set observed after the signal-date close."""

    signal_date: date
    rank_start: int
    rank_end: int
    symbols: tuple[Symbol, ...]
    target_count: int | None = None
    sell_buffer_rank: int | None = None

    def __post_init__(self) -> None:
        if self.rank_start < 1 or self.rank_end < self.rank_start:
            raise ValueError("micro-cap rank interval is invalid")
        expected = self.rank_end - self.rank_start + 1
        if len(self.symbols) != expected:
            raise ValueError(
                f"micro-cap rank target is incomplete: {len(self.symbols)} != {expected}"
            )
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("micro-cap rank target contains duplicate symbols")
        target_count = self.target_count or expected
        if target_count < 1 or target_count > expected:
            raise ValueError("micro-cap target count is invalid")
        if self.sell_buffer_rank is not None and not (
            target_count <= self.sell_buffer_rank <= expected
        ):
            raise ValueError("micro-cap sell buffer is invalid")
        object.__setattr__(self, "target_count", target_count)


@dataclass(frozen=True, slots=True)
class RankBandExecutionDiagnostic:
    signal_date: date
    target_count: int
    priced_count: int
    zero_lot_count: int
    retained_without_bar_count: int


class MicrocapRankBandStrategy:
    """Execute immutable PIT rank targets at T+1 open using whole board lots."""

    def __init__(
        self,
        *,
        targets: Mapping[date, MicrocapRankBandTarget],
        lot_size: int = 100,
        cash_buffer_weight: Decimal = Decimal("0.02"),
    ) -> None:
        buffer = Decimal(cash_buffer_weight)
        if not targets:
            raise ValueError("micro-cap rank strategy requires targets")
        if lot_size <= 0:
            raise ValueError("lot size must be positive")
        if not Decimal("0") <= buffer < Decimal("1"):
            raise ValueError("cash buffer weight must be in [0, 1)")
        if any(key != target.signal_date for key, target in targets.items()):
            raise ValueError("micro-cap target key and signal date disagree")
        self._targets = dict(targets)
        self._lot_size = lot_size
        self._cash_buffer = buffer
        self._diagnostics: list[RankBandExecutionDiagnostic] = []
        self._active_target: dict[Symbol, int] = {}

    @property
    def diagnostics(self) -> tuple[RankBandExecutionDiagnostic, ...]:
        return tuple(self._diagnostics)

    def on_close(self, context: StrategyContext) -> tuple[OrderRequest, ...]:
        trade_date = context.session.trade_date
        current = {position.symbol: position.quantity for position in context.portfolio.positions}
        self._apply_corporate_actions(context)
        target_spec = self._targets.get(trade_date)
        if target_spec is None:
            self._remove_hard_exits(context)
            return self._orders(current, self._active_target)

        bars = context.session.bar_by_symbol()
        selected = self._resolve_selected(target_spec, current)
        assert target_spec.target_count is not None
        investable = context.portfolio.equity * (Decimal("1") - self._cash_buffer)
        per_security = investable / target_spec.target_count
        target: dict[Symbol, int] = {}
        zero_lot_count = 0
        retained_without_bar_count = 0
        for symbol in selected:
            bar = bars.get(symbol)
            if bar is None:
                if current.get(symbol, 0):
                    target[symbol] = current[symbol]
                    retained_without_bar_count += 1
                continue
            quantity = self._round_lot(per_security / bar.close)
            if quantity:
                target[symbol] = quantity
            else:
                zero_lot_count += 1

        self._diagnostics.append(
            RankBandExecutionDiagnostic(
                signal_date=trade_date,
                target_count=target_spec.target_count,
                priced_count=sum(symbol in bars for symbol in selected),
                zero_lot_count=zero_lot_count,
                retained_without_bar_count=retained_without_bar_count,
            )
        )
        self._active_target = target
        return self._orders(current, self._active_target)

    @staticmethod
    def _orders(
        current: dict[Symbol, int],
        target: dict[Symbol, int],
    ) -> tuple[OrderRequest, ...]:
        orders = [
            OrderRequest(symbol, Side.BUY if delta > 0 else Side.SELL, abs(delta))
            for symbol in sorted(set(current) | set(target))
            if (delta := target.get(symbol, 0) - current.get(symbol, 0))
        ]
        orders.sort(key=lambda order: (order.side is Side.BUY, order.symbol))
        return tuple(orders)

    @staticmethod
    def _resolve_selected(
        target: MicrocapRankBandTarget,
        current: dict[Symbol, int],
    ) -> tuple[Symbol, ...]:
        assert target.target_count is not None
        if target.sell_buffer_rank is None:
            return target.symbols[: target.target_count]
        holding_zone = target.symbols[: target.sell_buffer_rank]
        retained = [symbol for symbol in holding_zone if current.get(symbol, 0)]
        retained = retained[: target.target_count]
        retained_set = set(retained)
        additions = [symbol for symbol in target.symbols if symbol not in retained_set][
            : target.target_count - len(retained)
        ]
        ordered_rank = {symbol: rank for rank, symbol in enumerate(target.symbols)}
        return tuple(sorted(retained + additions, key=ordered_rank.__getitem__))

    def _round_lot(self, quantity: Decimal) -> int:
        lots = (quantity / self._lot_size).to_integral_value(rounding=ROUND_FLOOR)
        return int(lots) * self._lot_size

    def _remove_hard_exits(
        self,
        context: StrategyContext,
    ) -> None:
        statuses = context.session.status_by_symbol()
        for symbol in tuple(self._active_target):
            if symbol in statuses and statuses[symbol].is_st:
                self._active_target.pop(symbol)

    def _apply_corporate_actions(self, context: StrategyContext) -> None:
        additive = {
            CorporateActionKind.STOCK_DIVIDEND,
            CorporateActionKind.CAPITALIZATION,
            CorporateActionKind.RIGHTS_ISSUE,
        }
        multiplicative = {
            CorporateActionKind.SPLIT,
            CorporateActionKind.REVERSE_SPLIT,
        }
        for action in context.session.corporate_actions:
            quantity = self._active_target.get(action.symbol)
            if quantity is None:
                continue
            if action.kind is CorporateActionKind.DELISTING:
                self._active_target.pop(action.symbol)
            elif action.kind in additive:
                self._active_target[action.symbol] = int(
                    Decimal(quantity) * (Decimal("1") + action.ratio)
                )
            elif action.kind in multiplicative:
                self._active_target[action.symbol] = int(Decimal(quantity) * action.ratio)
