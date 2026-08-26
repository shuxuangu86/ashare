import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from aquant.backtest.matching import Fill
from aquant.domain.corporate_actions import CorporateAction, CorporateActionKind
from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.domain.time import require_aware


@dataclass(frozen=True, slots=True)
class Position:
    symbol: Symbol
    quantity: int
    average_cost: Decimal

    def __post_init__(self) -> None:
        cost = Decimal(self.average_cost)
        if self.quantity < 0 or not cost.is_finite() or cost < 0:
            raise ValueError("position quantity/cost is invalid")
        if self.quantity == 0 and cost != 0:
            raise ValueError("zero position must have zero average cost")
        object.__setattr__(self, "average_cost", cost)


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    asof_time: datetime
    cash: Decimal
    market_value: Decimal
    equity: Decimal
    realized_pnl: Decimal
    positions: tuple[Position, ...]
    receivables: Decimal = Decimal("0")


@dataclass(frozen=True, slots=True)
class PortfolioLedgerState:
    initial_cash: Decimal
    cash: Decimal
    realized_pnl: Decimal
    receivables: Decimal
    dividend_receivables: tuple[tuple[UUID, Decimal], ...]
    positions: tuple[Position, ...]
    fill_ids: tuple[UUID, ...]
    action_ids: tuple[UUID, ...]
    last_fill_at: datetime | None
    last_event_at: datetime | None


class PortfolioLedger:
    """Fill-sourced accounting ledger; all state can be replayed from immutable fills."""

    def __init__(self, initial_cash: Decimal) -> None:
        cash = Decimal(initial_cash)
        if not cash.is_finite() or cash < 0:
            raise ValueError("initial cash must be finite and non-negative")
        self._initial_cash = cash
        self._cash = cash
        self._realized_pnl = Decimal("0")
        self._receivables = Decimal("0")
        self._dividend_receivables: dict[UUID, Decimal] = {}
        self._positions: dict[Symbol, Position] = {}
        self._fills: list[Fill] = []
        self._fill_ids: set[UUID] = set()
        self._action_ids: set[UUID] = set()
        self._actions: list[CorporateAction] = []
        self._last_fill_at: datetime | None = None
        self._last_event_at: datetime | None = None

    def validate_fill(self, fill: Fill) -> None:
        if fill.fill_id in self._fill_ids:
            raise ValueError(f"duplicate ledger fill id: {fill.fill_id}")
        if self._last_fill_at is not None and fill.occurred_at < self._last_fill_at:
            raise ValueError("ledger fills must be applied in chronological order")
        if self._last_event_at is not None and fill.occurred_at < self._last_event_at:
            raise ValueError("ledger events must be applied in chronological order")
        if fill.side is Side.BUY:
            required_cash = fill.notional + fill.fee
            if required_cash > self._cash:
                raise ValueError("insufficient cash for fill")
        else:
            position = self._positions.get(fill.symbol)
            if position is None or fill.quantity > position.quantity:
                raise ValueError("sell fill exceeds available position")

    def apply_fill(self, fill: Fill) -> None:
        self.validate_fill(fill)
        current = self._positions.get(fill.symbol, Position(fill.symbol, 0, Decimal("0")))
        if fill.side is Side.BUY:
            new_quantity = current.quantity + fill.quantity
            total_cost = current.average_cost * current.quantity + fill.notional + fill.fee
            self._positions[fill.symbol] = Position(
                fill.symbol, new_quantity, total_cost / new_quantity
            )
            self._cash -= fill.notional + fill.fee
        else:
            self._cash += fill.notional - fill.fee
            self._realized_pnl += (fill.price - current.average_cost) * fill.quantity - fill.fee
            remaining = current.quantity - fill.quantity
            self._positions[fill.symbol] = Position(
                fill.symbol,
                remaining,
                current.average_cost if remaining else Decimal("0"),
            )
        self._fills.append(fill)
        self._fill_ids.add(fill.fill_id)
        self._last_fill_at = fill.occurred_at
        self._last_event_at = fill.occurred_at

    def apply_corporate_action(self, action: CorporateAction) -> None:
        if action.action_id in self._action_ids:
            raise ValueError(f"duplicate corporate action id: {action.action_id}")
        if self._last_event_at is not None and action.occurred_at < self._last_event_at:
            raise ValueError("ledger events must be applied in chronological order")
        position = self.position(action.symbol)
        if action.kind is CorporateActionKind.CASH_DIVIDEND_PAYMENT:
            reference = action.reference_action_id
            if reference in self._dividend_receivables:
                amount = self._dividend_receivables.pop(reference)
                self._receivables -= amount
                self._cash += amount
        elif position.quantity:
            self._apply_position_action(action, position)
        self._actions.append(action)
        self._action_ids.add(action.action_id)
        self._last_event_at = action.occurred_at

    def _apply_position_action(self, action: CorporateAction, position: Position) -> None:
        if action.kind is CorporateActionKind.CASH_DIVIDEND_ENTITLEMENT:
            amount = action.cash_per_share * position.quantity
            self._dividend_receivables[action.action_id] = amount
            self._receivables += amount
            adjusted_cost = max(Decimal("0"), position.average_cost - action.cash_per_share)
            self._positions[action.symbol] = Position(
                action.symbol, position.quantity, adjusted_cost
            )
            return
        if action.kind is CorporateActionKind.DELISTING:
            proceeds = action.settlement_price * position.quantity
            self._cash += proceeds
            self._realized_pnl += proceeds - position.average_cost * position.quantity
            self._positions[action.symbol] = Position(action.symbol, 0, Decimal("0"))
            return
        multiplier = (
            action.ratio
            if action.kind in {CorporateActionKind.SPLIT, CorporateActionKind.REVERSE_SPLIT}
            else Decimal("1") + action.ratio
        )
        exact_quantity = Decimal(position.quantity) * multiplier
        new_quantity = int(exact_quantity)
        fractional = exact_quantity - new_quantity
        cash_in_lieu_price = action.cash_in_lieu_price
        if fractional and cash_in_lieu_price is None:
            raise ValueError("fractional corporate-action shares require cash-in-lieu price")
        if fractional:
            assert cash_in_lieu_price is not None
            self._cash += fractional * cash_in_lieu_price
        total_cost = position.average_cost * position.quantity
        if action.kind is CorporateActionKind.RIGHTS_ISSUE:
            subscribed = new_quantity - position.quantity
            subscription_cash = action.subscription_price * subscribed
            if subscription_cash > self._cash:
                raise ValueError("insufficient cash for rights-issue subscription")
            self._cash -= subscription_cash
            total_cost += subscription_cash
        self._positions[action.symbol] = Position(
            action.symbol,
            new_quantity,
            total_cost / new_quantity if new_quantity else Decimal("0"),
        )

    def snapshot(self, *, asof_time: datetime, prices: dict[Symbol, Decimal]) -> PortfolioSnapshot:
        timestamp = require_aware(asof_time, field_name="asof_time")
        market_value = Decimal("0")
        active_positions = tuple(
            sorted(
                (position for position in self._positions.values() if position.quantity),
                key=lambda item: item.symbol,
            )
        )
        for position in active_positions:
            try:
                price = Decimal(prices[position.symbol])
            except KeyError as exc:
                raise KeyError(f"missing mark price for {position.symbol}") from exc
            if not price.is_finite() or price <= 0:
                raise ValueError("mark prices must be finite and positive")
            market_value += price * position.quantity
        return PortfolioSnapshot(
            timestamp,
            self._cash,
            market_value,
            self._cash + market_value + self._receivables,
            self._realized_pnl,
            active_positions,
            self._receivables,
        )

    @classmethod
    def rebuild(
        cls,
        initial_cash: Decimal,
        fills: tuple[Fill, ...],
        actions: tuple[CorporateAction, ...] = (),
    ) -> "PortfolioLedger":
        ledger = cls(initial_cash)
        events: list[tuple[datetime, int, Fill | CorporateAction]] = [
            (fill.occurred_at, 1, fill) for fill in fills
        ]
        events.extend((action.occurred_at, 0, action) for action in actions)
        for _, _, event in sorted(events, key=lambda item: (item[0], item[1])):
            if isinstance(event, Fill):
                ledger.apply_fill(event)
            else:
                ledger.apply_corporate_action(event)
        return ledger

    @classmethod
    def from_state(cls, state: PortfolioLedgerState) -> "PortfolioLedger":
        ledger = cls(state.initial_cash)
        ledger._cash = Decimal(state.cash)
        ledger._realized_pnl = Decimal(state.realized_pnl)
        ledger._receivables = Decimal(state.receivables)
        ledger._dividend_receivables = dict(state.dividend_receivables)
        ledger._positions = {position.symbol: position for position in state.positions}
        ledger._fill_ids = set(state.fill_ids)
        ledger._action_ids = set(state.action_ids)
        ledger._last_fill_at = state.last_fill_at
        ledger._last_event_at = state.last_event_at
        return ledger

    @property
    def export_state(self) -> PortfolioLedgerState:
        return PortfolioLedgerState(
            initial_cash=self._initial_cash,
            cash=self._cash,
            realized_pnl=self._realized_pnl,
            receivables=self._receivables,
            dividend_receivables=tuple(sorted(self._dividend_receivables.items(), key=str)),
            positions=tuple(sorted(self._positions.values(), key=lambda item: item.symbol)),
            fill_ids=tuple(sorted(self._fill_ids, key=str)),
            action_ids=tuple(sorted(self._action_ids, key=str)),
            last_fill_at=self._last_fill_at,
            last_event_at=self._last_event_at,
        )

    @property
    def cash(self) -> Decimal:
        return self._cash

    def position(self, symbol: Symbol) -> Position:
        return self._positions.get(symbol, Position(symbol, 0, Decimal("0")))

    @property
    def realized_pnl(self) -> Decimal:
        return self._realized_pnl

    @property
    def receivables(self) -> Decimal:
        return self._receivables

    @property
    def fills(self) -> tuple[Fill, ...]:
        return tuple(self._fills)

    @property
    def corporate_actions(self) -> tuple[CorporateAction, ...]:
        return tuple(self._actions)

    @property
    def state_hash(self) -> str:
        payload = {
            "cash": str(self._cash),
            "corporate_actions": [
                {
                    "action_id": str(action.action_id),
                    "kind": action.kind.value,
                    "occurred_at": action.occurred_at.isoformat(),
                    "symbol": action.symbol.canonical,
                }
                for action in self._actions
            ],
            "fills": [
                {
                    "fee": str(fill.fee),
                    "fill_id": str(fill.fill_id),
                    "occurred_at": fill.occurred_at.isoformat(),
                    "order_id": str(fill.order_id),
                    "price": str(fill.price),
                    "quantity": fill.quantity,
                    "side": fill.side.value,
                    "symbol": fill.symbol.canonical,
                }
                for fill in self._fills
            ],
            "initial_cash": str(self._initial_cash),
            "realized_pnl": str(self._realized_pnl),
            "receivables": str(self._receivables),
            "positions": [
                {
                    "average_cost": str(position.average_cost),
                    "quantity": position.quantity,
                    "symbol": position.symbol.canonical,
                }
                for position in sorted(self._positions.values(), key=lambda item: item.symbol)
                if position.quantity
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()
