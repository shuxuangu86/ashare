"""Vectorized factor analysis and event-driven strategy backtesting."""

from aquant.backtest.accounting import PortfolioLedger, PortfolioSnapshot, Position
from aquant.backtest.event_engine import EventKind
from aquant.backtest.event_engine.engine import (
    BacktestResult,
    EventDrivenBacktest,
    EventDrivenStrategy,
    MarketSession,
    StrategyContext,
)
from aquant.backtest.matching import (
    BacktestOrder,
    BacktestOrderStatus,
    Fill,
    NextOpenMatcher,
    OrderRequest,
)

__all__ = [
    "BacktestOrder",
    "BacktestOrderStatus",
    "BacktestResult",
    "EventDrivenBacktest",
    "EventDrivenStrategy",
    "EventKind",
    "Fill",
    "MarketSession",
    "NextOpenMatcher",
    "OrderRequest",
    "PortfolioLedger",
    "PortfolioSnapshot",
    "Position",
    "StrategyContext",
]
