import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_FLOOR, Decimal
from itertools import pairwise
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from aquant.backtest.accounting import PortfolioLedger, PortfolioSnapshot
from aquant.backtest.event_engine.events import (
    BacktestEvent,
    EventPriority,
    SessionCloseEvent,
    SessionOpenEvent,
)
from aquant.backtest.event_engine.queue import EventQueue
from aquant.backtest.matching import (
    BacktestOrder,
    Fill,
    FillEvent,
    NextOpenMatcher,
    OrderBook,
    OrderRequest,
    OrderSubmittedEvent,
)
from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar, SecurityStatus
from aquant.domain.time import require_aware


@dataclass(frozen=True, slots=True)
class MarketSession:
    trade_date: date
    open_at: datetime
    close_at: datetime
    bars: tuple[DailyBar, ...]
    statuses: tuple[SecurityStatus, ...] = ()

    def __post_init__(self) -> None:
        opened = require_aware(self.open_at, field_name="open_at")
        closed = require_aware(self.close_at, field_name="close_at")
        if opened >= closed or opened.date() != self.trade_date or closed.date() != self.trade_date:
            raise ValueError("market session timestamps must be ordered on trade_date")
        symbols = [bar.symbol for bar in self.bars]
        if not self.bars or len(symbols) != len(set(symbols)):
            raise ValueError("market session bars must be non-empty with unique symbols")
        if any(bar.trade_date != self.trade_date for bar in self.bars):
            raise ValueError("market session bar date mismatch")
        status_symbols = [status.symbol for status in self.statuses]
        if len(status_symbols) != len(set(status_symbols)) or any(
            status.trade_date != self.trade_date for status in self.statuses
        ):
            raise ValueError("market session security statuses are invalid")
        object.__setattr__(self, "open_at", opened)
        object.__setattr__(self, "close_at", closed)

    def bar_by_symbol(self) -> dict[Symbol, DailyBar]:
        return {bar.symbol: bar for bar in self.bars}

    def status_by_symbol(self) -> dict[Symbol, SecurityStatus]:
        return {status.symbol: status for status in self.statuses}


@dataclass(frozen=True, slots=True)
class StrategyContext:
    session: MarketSession
    portfolio: PortfolioSnapshot


class EventDrivenStrategy(Protocol):
    def on_close(self, context: StrategyContext) -> tuple[OrderRequest, ...]: ...


@dataclass(frozen=True, slots=True)
class BacktestResult:
    run_id: str
    events: tuple[BacktestEvent, ...]
    orders: tuple[BacktestOrder, ...]
    fills: tuple[Fill, ...]
    equity_curve: tuple[PortfolioSnapshot, ...]
    final_state_hash: str


SessionEvent = SessionOpenEvent | SessionCloseEvent


class OpenMatcher(Protocol):
    def match(
        self,
        order: BacktestOrder,
        bar: DailyBar,
        *,
        occurred_at: datetime,
        fill_id: UUID,
        maximum_quantity: int | None = None,
        available_cash: Decimal | None = None,
        status: SecurityStatus | None = None,
    ) -> Fill | None: ...


class EventDrivenBacktest:
    def __init__(
        self,
        *,
        run_id: str,
        initial_cash: Decimal,
        matcher: OpenMatcher | None = None,
    ) -> None:
        if not run_id.strip():
            raise ValueError("backtest run_id must not be blank")
        self._run_id = run_id.strip()
        self._initial_cash = Decimal(initial_cash)
        self._matcher = matcher or NextOpenMatcher()

    def run(
        self,
        sessions: tuple[MarketSession, ...],
        strategy: EventDrivenStrategy,
    ) -> BacktestResult:
        self._validate_sessions(sessions)
        queue: EventQueue[SessionEvent] = EventQueue()
        session_by_date = {session.trade_date: session for session in sessions}
        for session in sessions:
            queue.push(
                SessionOpenEvent(session.trade_date, session.open_at),
                occurred_at=session.open_at,
                priority=EventPriority.SESSION_OPEN,
            )
            queue.push(
                SessionCloseEvent(session.trade_date, session.close_at),
                occurred_at=session.close_at,
                priority=EventPriority.SESSION_CLOSE,
            )

        order_book = OrderBook()
        ledger = PortfolioLedger(self._initial_cash)
        events: list[BacktestEvent] = []
        equity_curve: list[PortfolioSnapshot] = []
        order_sequence = 0
        fill_sequence = 0

        while queue:
            scheduled = queue.pop()
            event = scheduled.event
            session = session_by_date[event.trade_date]
            events.append(event)
            if isinstance(event, SessionOpenEvent):
                bars = session.bar_by_symbol()
                statuses = session.status_by_symbol()
                for order in order_book.eligible_orders(event.occurred_at):
                    bar = bars.get(order.symbol)
                    if bar is None:
                        continue
                    maximum = self._maximum_executable(order, bar.open, ledger)
                    fill_sequence += 1
                    fill = self._matcher.match(
                        order,
                        bar,
                        occurred_at=event.occurred_at,
                        fill_id=self._identifier("fill", fill_sequence),
                        maximum_quantity=maximum,
                        available_cash=ledger.cash,
                        status=statuses.get(order.symbol),
                    )
                    if fill is None:
                        continue
                    order_book.validate_fill(fill)
                    ledger.validate_fill(fill)
                    order_book.apply_fill(fill)
                    ledger.apply_fill(fill)
                    events.append(FillEvent(fill))
                continue

            prices = {bar.symbol: bar.close for bar in session.bars}
            snapshot = ledger.snapshot(asof_time=event.occurred_at, prices=prices)
            equity_curve.append(snapshot)
            context = StrategyContext(session, snapshot)
            for request in strategy.on_close(context):
                order_sequence += 1
                order = BacktestOrder(
                    order_id=self._identifier("order", order_sequence),
                    symbol=request.symbol,
                    side=request.side,
                    quantity=request.quantity,
                    submitted_at=event.occurred_at,
                )
                order_book.submit(order)
                events.append(OrderSubmittedEvent(order))

        final_hash = self._result_hash(order_book.orders, ledger, equity_curve)
        return BacktestResult(
            self._run_id,
            tuple(events),
            order_book.orders,
            ledger.fills,
            tuple(equity_curve),
            final_hash,
        )

    def _identifier(self, kind: str, sequence: int) -> UUID:
        return uuid5(NAMESPACE_URL, f"aquant:{self._run_id}:{kind}:{sequence}")

    @staticmethod
    def _maximum_executable(order: BacktestOrder, price: Decimal, ledger: PortfolioLedger) -> int:
        if order.side is Side.SELL:
            return ledger.position(order.symbol).quantity
        return int((ledger.cash / price).to_integral_value(rounding=ROUND_FLOOR))

    @staticmethod
    def _validate_sessions(sessions: tuple[MarketSession, ...]) -> None:
        if not sessions:
            raise ValueError("event-driven backtest requires market sessions")
        for previous, current in pairwise(sessions):
            if previous.trade_date >= current.trade_date or previous.close_at >= current.open_at:
                raise ValueError(
                    "market sessions must be strictly chronological and non-overlapping"
                )

    def _result_hash(
        self,
        orders: tuple[BacktestOrder, ...],
        ledger: PortfolioLedger,
        equity_curve: list[PortfolioSnapshot],
    ) -> str:
        payload = {
            "equity": [str(snapshot.equity) for snapshot in equity_curve],
            "ledger": ledger.state_hash,
            "orders": [
                {
                    "filled": order.filled_quantity,
                    "id": str(order.order_id),
                    "quantity": order.quantity,
                    "side": order.side.value,
                    "status": order.status.value,
                    "symbol": order.symbol.canonical,
                }
                for order in orders
            ],
            "run_id": self._run_id,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
