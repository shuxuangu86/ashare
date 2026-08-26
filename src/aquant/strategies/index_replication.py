from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_FLOOR, Decimal

from aquant.backtest.event_engine.engine import StrategyContext
from aquant.backtest.matching import OrderRequest
from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol


@dataclass(frozen=True, slots=True)
class IndexRebalanceTarget:
    signal_date: date
    effective_date: date
    weights: tuple[tuple[Symbol, Decimal], ...]
    execution_prices: tuple[tuple[Symbol, Decimal], ...]
    source_snapshot_date: date
    reason: str
    portfolio_valuation_prices: tuple[tuple[Symbol, Decimal], ...] = ()

    def __post_init__(self) -> None:
        weight_symbols = [symbol for symbol, _ in self.weights]
        price_symbols = [symbol for symbol, _ in self.execution_prices]
        total = sum((weight for _, weight in self.weights), Decimal("0"))
        if self.signal_date >= self.effective_date:
            raise ValueError("index target signal must precede its effective date")
        if not self.weights or weight_symbols != price_symbols:
            raise ValueError("index target weights and execution prices must align")
        if len(weight_symbols) != len(set(weight_symbols)):
            raise ValueError("index target symbols must be unique")
        if any(weight <= 0 for _, weight in self.weights):
            raise ValueError("index target weights must be positive")
        if any(price <= 0 for _, price in self.execution_prices):
            raise ValueError("index target execution prices must be positive")
        valuation_symbols = [symbol for symbol, _ in self.portfolio_valuation_prices]
        if len(valuation_symbols) != len(set(valuation_symbols)) or any(
            price <= 0 for _, price in self.portfolio_valuation_prices
        ):
            raise ValueError("index target portfolio valuation prices are invalid")
        if abs(total - Decimal("1")) > Decimal("0.000001"):
            raise ValueError(f"index target weights must sum to one: {total}")
        if not self.reason.strip():
            raise ValueError("index target reason must not be blank")


class IndexReplicationStrategy:
    """Calibration-only weighted index replication with next-open execution."""

    def __init__(
        self,
        targets: Mapping[date, IndexRebalanceTarget],
        *,
        cash_buffer_weight: Decimal = Decimal("0.0001"),
    ) -> None:
        cash_buffer = Decimal(cash_buffer_weight)
        if not Decimal("0") <= cash_buffer < Decimal("1"):
            raise ValueError("index replication cash buffer must be in [0, 1)")
        self._targets = dict(targets)
        if not self._targets or any(
            day != target.signal_date for day, target in self._targets.items()
        ):
            raise ValueError("index replication targets must be non-empty and keyed by signal date")
        self._cash_buffer = cash_buffer

    def on_close(self, context: StrategyContext) -> tuple[OrderRequest, ...]:
        target = self._targets.get(context.session.trade_date)
        if target is None:
            return ()
        current = {position.symbol: position.quantity for position in context.portfolio.positions}
        valuation_prices = dict(target.portfolio_valuation_prices)
        effective_equity = context.portfolio.equity
        if valuation_prices:
            effective_equity = (
                context.portfolio.cash
                + context.portfolio.receivables
                + sum(
                    (
                        Decimal(quantity) * valuation_prices[symbol]
                        for symbol, quantity in current.items()
                    ),
                    Decimal("0"),
                )
            )
        investable = effective_equity * (Decimal("1") - self._cash_buffer)
        target_quantities = {
            symbol: int((investable * weight / price).to_integral_value(rounding=ROUND_FLOOR))
            for (symbol, weight), (_, price) in zip(
                target.weights,
                target.execution_prices,
                strict=True,
            )
        }
        orders = [
            OrderRequest(symbol, Side.BUY if delta > 0 else Side.SELL, abs(delta))
            for symbol in sorted(set(current) | set(target_quantities))
            if (delta := target_quantities.get(symbol, 0) - current.get(symbol, 0))
        ]
        orders.sort(key=lambda order: (order.side is Side.BUY, order.symbol))
        return tuple(orders)
