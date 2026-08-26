from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal

from aquant.backtest import BacktestResult, MarketSession
from aquant.backtest.matching import Fill


@dataclass(frozen=True, slots=True)
class MonthlyPortfolioResult:
    month: str
    net_return: Decimal
    turnover: Decimal
    fees: Decimal
    slippage: Decimal
    friction_cost: Decimal
    friction_bps: Decimal
    ending_equity: Decimal

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class MonthlyBenchmarkResult:
    month: str
    return_rate: Decimal
    ending_close: Decimal

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def monthly_portfolio_results(
    result: BacktestResult,
    *,
    initial_equity: Decimal,
    sessions: Sequence[MarketSession],
) -> tuple[MonthlyPortfolioResult, ...]:
    if not result.equity_curve:
        raise ValueError("monthly portfolio report requires an equity curve")
    session_by_date = {session.trade_date: session for session in sessions}
    month_end_equity: dict[str, Decimal] = {}
    for snapshot in result.equity_curve:
        month_end_equity[snapshot.asof_time.strftime("%Y-%m")] = snapshot.equity
    fills_by_month: dict[str, list[Fill]] = {}
    for fill in result.fills:
        fills_by_month.setdefault(fill.occurred_at.strftime("%Y-%m"), []).append(fill)

    previous_equity = Decimal(initial_equity)
    rows: list[MonthlyPortfolioResult] = []
    for month, ending_equity in sorted(month_end_equity.items()):
        fills = fills_by_month.get(month, [])
        fees = sum((fill.fee for fill in fills), Decimal("0"))
        notional = sum((fill.notional for fill in fills), Decimal("0"))
        slippage = Decimal("0")
        for fill in fills:
            session = session_by_date[fill.occurred_at.date()]
            reference_open = session.bar_by_symbol()[fill.symbol].open
            slippage += abs(fill.price - reference_open) * fill.quantity
        friction = fees + slippage
        rows.append(
            MonthlyPortfolioResult(
                month=month,
                net_return=ending_equity / previous_equity - Decimal("1"),
                turnover=notional / previous_equity,
                fees=fees,
                slippage=slippage,
                friction_cost=friction,
                friction_bps=friction / previous_equity * Decimal("10000"),
                ending_equity=ending_equity,
            )
        )
        previous_equity = ending_equity
    return tuple(rows)


def monthly_benchmark_results(
    closes: Mapping[date, Decimal],
    *,
    start_date: date,
    end_date: date,
) -> tuple[MonthlyBenchmarkResult, ...]:
    ordered = sorted((day, Decimal(value)) for day, value in closes.items())
    prior = [value for day, value in ordered if day < start_date]
    if not prior:
        raise ValueError("benchmark requires one close before the comparison period")
    previous_close = prior[-1]
    month_ends: dict[str, Decimal] = {}
    for day, value in ordered:
        if start_date <= day <= end_date:
            month_ends[day.strftime("%Y-%m")] = value
    if not month_ends:
        raise ValueError("benchmark has no closes in the comparison period")
    results: list[MonthlyBenchmarkResult] = []
    for month, ending_close in sorted(month_ends.items()):
        results.append(
            MonthlyBenchmarkResult(
                month=month,
                return_rate=ending_close / previous_close - Decimal("1"),
                ending_close=ending_close,
            )
        )
        previous_close = ending_close
    return tuple(results)
