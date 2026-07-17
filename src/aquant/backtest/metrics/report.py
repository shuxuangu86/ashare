from dataclasses import dataclass
from decimal import Decimal

from aquant.backtest.event_engine.engine import BacktestResult


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    initial_equity: Decimal
    final_equity: Decimal
    total_return: Decimal
    max_drawdown: Decimal
    gross_turnover: Decimal
    total_fees: Decimal
    fill_rate: Decimal
    trade_count: int


def calculate_metrics(result: BacktestResult, *, initial_equity: Decimal) -> BacktestMetrics:
    initial = Decimal(initial_equity)
    if initial <= 0 or not result.equity_curve:
        raise ValueError("metrics require positive initial equity and an equity curve")
    equities = [snapshot.equity for snapshot in result.equity_curve]
    peak = equities[0]
    maximum_drawdown = Decimal("0")
    for equity in equities:
        peak = max(peak, equity)
        if peak > 0:
            maximum_drawdown = max(maximum_drawdown, (peak - equity) / peak)
    gross_notional = sum((fill.notional for fill in result.fills), Decimal("0"))
    total_ordered = sum(order.quantity for order in result.orders)
    total_filled = sum(fill.quantity for fill in result.fills)
    return BacktestMetrics(
        initial,
        equities[-1],
        equities[-1] / initial - Decimal("1"),
        maximum_drawdown,
        gross_notional / initial,
        sum((fill.fee for fill in result.fills), Decimal("0")),
        Decimal(total_filled) / total_ordered if total_ordered else Decimal("0"),
        len(result.fills),
    )


def render_markdown_report(result: BacktestResult, *, initial_equity: Decimal) -> str:
    metrics = calculate_metrics(result, initial_equity=initial_equity)
    return f"""# Backtest Report: {result.run_id}

- Initial equity: {metrics.initial_equity}
- Final equity: {metrics.final_equity}
- Total return: {metrics.total_return:.4%}
- Maximum drawdown: {metrics.max_drawdown:.4%}
- Gross turnover: {metrics.gross_turnover:.4f}
- Total fees: {metrics.total_fees}
- Fill rate: {metrics.fill_rate:.2%}
- Trade count: {metrics.trade_count}
- Final state SHA-256: `{result.final_state_hash}`
"""
