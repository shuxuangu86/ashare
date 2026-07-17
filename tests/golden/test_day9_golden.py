import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from aquant.backtest import EventDrivenBacktest, MarketSession, OrderRequest, StrategyContext
from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar

SYMBOL = Symbol.parse("600000.XSHG")
EXPECTED = Path(__file__).parent / "fixtures" / "day9_expected.json"


def _session(day: int, opening: str, close: str) -> MarketSession:
    trade_date = date(2026, 7, day)
    open_price = Decimal(opening)
    close_price = Decimal(close)
    bar = DailyBar(
        SYMBOL,
        trade_date,
        open_price,
        max(open_price, close_price),
        min(open_price, close_price),
        close_price,
        Decimal("1000000"),
        Decimal("10000000"),
    )
    return MarketSession(
        trade_date,
        datetime(2026, 7, day, 1, 30, tzinfo=UTC),
        datetime(2026, 7, day, 7, tzinfo=UTC),
        (bar,),
    )


class GoldenStrategy:
    def on_close(self, context: StrategyContext) -> tuple[OrderRequest, ...]:
        if context.session.trade_date.day == 15:
            return (OrderRequest(SYMBOL, Side.BUY, 100),)
        if context.session.trade_date.day == 16:
            return (OrderRequest(SYMBOL, Side.SELL, 100),)
        return ()


def test_day9_event_and_accounting_golden_result() -> None:
    expected = json.loads(EXPECTED.read_text())
    result = EventDrivenBacktest(run_id="golden-day9", initial_cash=Decimal("10000")).run(
        (
            _session(15, "10", "10"),
            _session(16, "11", "12"),
            _session(17, "13", "13"),
        ),
        GoldenStrategy(),
    )
    actual = {
        "equity_curve": [str(snapshot.equity) for snapshot in result.equity_curve],
        "event_kinds": [event.kind.value for event in result.events],
        "fills": [
            {"price": str(fill.price), "quantity": fill.quantity, "side": fill.side.value}
            for fill in result.fills
        ],
        "final_state_hash": result.final_state_hash,
    }
    assert actual == expected
