import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from aquant.backtest.matching import Fill
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


class PortfolioLedger:
    """Fill-sourced accounting ledger; all state can be replayed from immutable fills."""

    def __init__(self, initial_cash: Decimal) -> None:
        cash = Decimal(initial_cash)
        if not cash.is_finite() or cash < 0:
            raise ValueError("initial cash must be finite and non-negative")
        self._initial_cash = cash
        self._cash = cash
        self._realized_pnl = Decimal("0")
        self._positions: dict[Symbol, Position] = {}
        self._fills: list[Fill] = []
        self._fill_ids: set[UUID] = set()
        self._last_fill_at: datetime | None = None

    def validate_fill(self, fill: Fill) -> None:
        if fill.fill_id in self._fill_ids:
            raise ValueError(f"duplicate ledger fill id: {fill.fill_id}")
        if self._last_fill_at is not None and fill.occurred_at < self._last_fill_at:
            raise ValueError("ledger fills must be applied in chronological order")
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
            self._cash + market_value,
            self._realized_pnl,
            active_positions,
        )

    @classmethod
    def rebuild(cls, initial_cash: Decimal, fills: tuple[Fill, ...]) -> "PortfolioLedger":
        ledger = cls(initial_cash)
        for fill in fills:
            ledger.apply_fill(fill)
        return ledger

    @property
    def cash(self) -> Decimal:
        return self._cash

    def position(self, symbol: Symbol) -> Position:
        return self._positions.get(symbol, Position(symbol, 0, Decimal("0")))

    @property
    def realized_pnl(self) -> Decimal:
        return self._realized_pnl

    @property
    def fills(self) -> tuple[Fill, ...]:
        return tuple(self._fills)

    @property
    def state_hash(self) -> str:
        payload = {
            "cash": str(self._cash),
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
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()
