from datetime import UTC, date, datetime
from decimal import Decimal

from aquant.backtest import BacktestResult, MarketSession
from aquant.backtest.accounting import PortfolioSnapshot
from aquant.backtest.matching import Fill
from aquant.domain.enums import Exchange, Side
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar
from aquant.strategies import monthly_benchmark_results, monthly_portfolio_results


def test_monthly_benchmark_uses_pre_period_close_for_first_month() -> None:
    rows = monthly_benchmark_results(
        {
            date(2025, 12, 31): Decimal("100"),
            date(2026, 1, 30): Decimal("110"),
            date(2026, 2, 27): Decimal("99"),
        },
        start_date=date(2026, 1, 1),
        end_date=date(2026, 2, 28),
    )
    assert [row.return_rate for row in rows] == [Decimal("0.1"), Decimal("-0.1")]


def test_monthly_portfolio_reports_net_return_turnover_and_friction() -> None:
    symbol = Symbol("000001", Exchange.XSHE)
    session = MarketSession(
        date(2026, 1, 30),
        datetime(2026, 1, 30, 1, 30, tzinfo=UTC),
        datetime(2026, 1, 30, 7, 0, tzinfo=UTC),
        (
            DailyBar(
                symbol,
                date(2026, 1, 30),
                Decimal("10"),
                Decimal("11"),
                Decimal("9"),
                Decimal("10"),
                Decimal("100000"),
                Decimal("1000000"),
            ),
        ),
    )
    fill = Fill(
        __import__("uuid").uuid4(),
        __import__("uuid").uuid4(),
        symbol,
        Side.BUY,
        100,
        Decimal("10.005"),
        Decimal("5"),
        session.open_at,
    )
    snapshot = PortfolioSnapshot(
        session.close_at,
        Decimal("0"),
        Decimal("110"),
        Decimal("110"),
        Decimal("0"),
        (),
    )
    result = BacktestResult("run", (), (), (fill,), (snapshot,), "hash")
    row = monthly_portfolio_results(
        result,
        initial_equity=Decimal("100"),
        sessions=(session,),
    )[0]
    assert row.net_return == Decimal("0.1")
    assert row.turnover == Decimal("10.005")
    assert row.slippage == Decimal("0.500")
    assert row.friction_cost == Decimal("5.500")
