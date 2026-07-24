"""Vectorized factor analysis and event-driven strategy backtesting."""

from aquant.backtest.accounting import PortfolioLedger, PortfolioSnapshot, Position
from aquant.backtest.costs import AshareFeeModel, AshareFeeSchedule, FeeBreakdown
from aquant.backtest.event_engine import EventKind
from aquant.backtest.event_engine.engine import (
    BacktestResult,
    EventDrivenBacktest,
    EventDrivenStrategy,
    MarketSession,
    StrategyContext,
    UnfilledOrderPolicy,
)
from aquant.backtest.matching import (
    AshareExecutionRules,
    AshareOpenMatcher,
    BacktestOrder,
    BacktestOrderStatus,
    Fill,
    NextOpenMatcher,
    OrderRequest,
)
from aquant.backtest.metrics import BacktestMetrics, calculate_metrics, render_markdown_report

__all__ = [
    "AshareExecutionRules",
    "AshareFeeModel",
    "AshareFeeSchedule",
    "AshareOpenMatcher",
    "BacktestMetrics",
    "BacktestOrder",
    "BacktestOrderStatus",
    "BacktestResult",
    "EventDrivenBacktest",
    "EventDrivenStrategy",
    "EventKind",
    "FeeBreakdown",
    "Fill",
    "MarketSession",
    "NextOpenMatcher",
    "OrderRequest",
    "PortfolioLedger",
    "PortfolioSnapshot",
    "Position",
    "StrategyContext",
    "UnfilledOrderPolicy",
    "calculate_metrics",
    "render_markdown_report",
]
