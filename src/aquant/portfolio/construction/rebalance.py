import hashlib
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.domain.orders import build_idempotency_key
from aquant.domain.portfolio import TargetPortfolio
from aquant.execution.broker_api import BrokerOrderRequest


@dataclass(frozen=True, slots=True)
class RebalancePlan:
    strategy_id: str
    data_release_id: str
    signal_version: str
    orders: tuple[BrokerOrderRequest, ...]


class RebalancePlanner:
    def plan(
        self,
        target: TargetPortfolio,
        *,
        account_id: str,
        equity: Decimal,
        current_quantities: dict[Symbol, int],
        prices: dict[Symbol, Decimal],
        lot_size: int = 100,
    ) -> RebalancePlan:
        if not account_id.strip() or Decimal(equity) <= 0 or lot_size <= 0:
            raise ValueError("rebalance account/equity/lot inputs are invalid")
        target_weights = {position.symbol: position.target_weight for position in target.positions}
        universe = set(target_weights) | set(current_quantities)
        planned: list[BrokerOrderRequest] = []
        for symbol in sorted(universe):
            try:
                price = Decimal(prices[symbol])
            except KeyError as exc:
                raise KeyError(f"missing rebalance price for {symbol}") from exc
            if price <= 0:
                raise ValueError("rebalance prices must be positive")
            target_value = Decimal(equity) * target_weights.get(symbol, Decimal("0"))
            target_quantity = (
                int((target_value / price / lot_size).to_integral_value(rounding=ROUND_FLOOR))
                * lot_size
            )
            delta = target_quantity - current_quantities.get(symbol, 0)
            if delta == 0:
                continue
            side = Side.BUY if delta > 0 else Side.SELL
            quantity = abs(delta)
            idempotency_key = build_idempotency_key(
                account_id=account_id,
                strategy_id=target.strategy_id,
                trade_date=target.trade_date,
                symbol=symbol,
                side=side,
                target_quantity=target_quantity,
            )
            client_id = hashlib.sha256(
                f"{target.signal_version}:{idempotency_key}".encode()
            ).hexdigest()[:24]
            planned.append(BrokerOrderRequest(client_id, idempotency_key, symbol, side, quantity))
        planned.sort(key=lambda order: (order.side is Side.BUY, order.symbol))
        return RebalancePlan(
            target.strategy_id,
            target.data_release_id,
            target.signal_version,
            tuple(planned),
        )
