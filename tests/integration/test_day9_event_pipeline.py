from datetime import UTC, date, datetime
from decimal import Decimal

from aquant.backtest import EventDrivenBacktest, MarketSession, OrderRequest, StrategyContext
from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar
from aquant.factors import ExpressionEvaluator, FactorPanel, parse_expression

SYMBOLS = (Symbol.parse("600000.XSHG"), Symbol.parse("000001.XSHE"))


def _session(day: int, closes: tuple[str, str]) -> MarketSession:
    trade_date = date(2026, 7, day)
    bars = tuple(
        DailyBar(
            symbol,
            trade_date,
            Decimal(close),
            Decimal(close),
            Decimal(close),
            Decimal(close),
            Decimal("100000"),
            Decimal("1000000"),
        )
        for symbol, close in zip(SYMBOLS, closes, strict=True)
    )
    return MarketSession(
        trade_date,
        datetime(2026, 7, day, 1, 30, tzinfo=UTC),
        datetime(2026, 7, day, 7, tzinfo=UTC),
        bars,
    )


class FactorRankStrategy:
    """Integration seam: tested DSL output becomes an event-engine order request."""

    def on_close(self, context: StrategyContext) -> tuple[OrderRequest, ...]:
        if context.session.trade_date != date(2026, 7, 15):
            return ()
        closes = [[float(bar.close) for bar in context.session.bars]]
        panel = FactorPanel(tuple(str(symbol) for symbol in SYMBOLS), {"Close": closes})
        ranks = ExpressionEvaluator().evaluate(parse_expression("RankCS(Close)"), panel)
        winner = SYMBOLS[int(ranks[0].argmax())]
        return (OrderRequest(winner, Side.BUY, 100),)


def test_factor_signal_to_next_open_fill_to_accounting_pipeline() -> None:
    sessions = (
        _session(15, ("10", "20")),
        _session(16, ("11", "21")),
    )
    result = EventDrivenBacktest(run_id="factor-to-accounting", initial_cash=Decimal("10000")).run(
        sessions, FactorRankStrategy()
    )

    assert len(result.fills) == 1
    assert result.fills[0].symbol == SYMBOLS[1]
    assert result.fills[0].occurred_at == sessions[1].open_at
    assert result.equity_curve[-1].positions[0].symbol == SYMBOLS[1]
    assert result.equity_curve[-1].equity == Decimal("10000")
