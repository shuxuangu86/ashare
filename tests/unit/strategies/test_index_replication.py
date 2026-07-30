from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from aquant.backtest import EventDrivenBacktest, MarketSession
from aquant.domain.enums import Exchange
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar
from aquant.strategies import IndexRebalanceTarget, IndexReplicationStrategy

SYMBOLS = (
    Symbol("600000", Exchange.XSHG),
    Symbol("000001", Exchange.XSHE),
)


def _session(trade_date: date, prices: tuple[str, str]) -> MarketSession:
    return MarketSession(
        trade_date,
        datetime.combine(trade_date, datetime.min.time(), UTC).replace(hour=1, minute=30),
        datetime.combine(trade_date, datetime.min.time(), UTC).replace(hour=7),
        tuple(
            DailyBar(
                symbol,
                trade_date,
                Decimal(price),
                Decimal(price),
                Decimal(price),
                Decimal(price),
                Decimal("1000000"),
                Decimal("10000000"),
            )
            for symbol, price in zip(SYMBOLS, prices, strict=True)
        ),
    )


def test_index_replication_rebalances_at_next_open_without_costs() -> None:
    signal = date(2024, 1, 2)
    effective = date(2024, 1, 3)
    target = IndexRebalanceTarget(
        signal,
        effective,
        ((SYMBOLS[0], Decimal("0.6")), (SYMBOLS[1], Decimal("0.4"))),
        ((SYMBOLS[0], Decimal("10")), (SYMBOLS[1], Decimal("20"))),
        signal,
        "INITIAL",
    )
    result = EventDrivenBacktest(
        run_id="index-replication",
        initial_cash=Decimal("100000"),
    ).run(
        (_session(signal, ("10", "20")), _session(effective, ("10", "20"))),
        IndexReplicationStrategy({signal: target}, cash_buffer_weight=Decimal("0")),
    )
    assert {fill.symbol: fill.quantity for fill in result.fills} == {
        SYMBOLS[0]: 6000,
        SYMBOLS[1]: 2000,
    }
    assert all(fill.fee == 0 for fill in result.fills)
    assert result.equity_curve[-1].equity == Decimal("100000")


@pytest.mark.parametrize(
    "weights,prices",
    [
        (((SYMBOLS[0], Decimal("0.9")),), ((SYMBOLS[0], Decimal("10")),)),
        (
            ((SYMBOLS[0], Decimal("0.5")), (SYMBOLS[0], Decimal("0.5"))),
            ((SYMBOLS[0], Decimal("10")), (SYMBOLS[0], Decimal("10"))),
        ),
    ],
)
def test_index_target_rejects_invalid_weights(
    weights: tuple[tuple[Symbol, Decimal], ...],
    prices: tuple[tuple[Symbol, Decimal], ...],
) -> None:
    with pytest.raises(ValueError):
        IndexRebalanceTarget(
            date(2024, 1, 2),
            date(2024, 1, 3),
            weights,
            prices,
            date(2024, 1, 2),
            "TEST",
        )


def test_index_target_and_strategy_reject_invalid_configuration() -> None:
    valid_weights = ((SYMBOLS[0], Decimal("1")),)
    valid_prices = ((SYMBOLS[0], Decimal("10")),)
    with pytest.raises(ValueError, match="must precede"):
        IndexRebalanceTarget(
            date(2024, 1, 3),
            date(2024, 1, 3),
            valid_weights,
            valid_prices,
            date(2024, 1, 2),
            "TEST",
        )
    with pytest.raises(ValueError, match="must align"):
        IndexRebalanceTarget(
            date(2024, 1, 2),
            date(2024, 1, 3),
            valid_weights,
            ((SYMBOLS[1], Decimal("10")),),
            date(2024, 1, 2),
            "TEST",
        )
    with pytest.raises(ValueError, match="valuation prices"):
        IndexRebalanceTarget(
            date(2024, 1, 2),
            date(2024, 1, 3),
            valid_weights,
            valid_prices,
            date(2024, 1, 2),
            "TEST",
            ((SYMBOLS[0], Decimal("0")),),
        )
    with pytest.raises(ValueError, match="reason"):
        IndexRebalanceTarget(
            date(2024, 1, 2),
            date(2024, 1, 3),
            valid_weights,
            valid_prices,
            date(2024, 1, 2),
            " ",
        )
    with pytest.raises(ValueError, match="cash buffer"):
        IndexReplicationStrategy(
            {
                date(2024, 1, 2): IndexRebalanceTarget(
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    valid_weights,
                    valid_prices,
                    date(2024, 1, 2),
                    "TEST",
                )
            },
            cash_buffer_weight=Decimal("1"),
        )
    with pytest.raises(ValueError, match="keyed by signal date"):
        IndexReplicationStrategy({})


def test_index_replication_uses_effective_open_portfolio_valuation() -> None:
    signal = date(2024, 1, 2)
    effective = date(2024, 1, 3)
    target = IndexRebalanceTarget(
        signal,
        effective,
        ((SYMBOLS[1], Decimal("1")),),
        ((SYMBOLS[1], Decimal("20")),),
        signal,
        "REPLACE",
        ((SYMBOLS[0], Decimal("20")),),
    )
    initial = IndexRebalanceTarget(
        date(2024, 1, 1),
        signal,
        ((SYMBOLS[0], Decimal("1")),),
        ((SYMBOLS[0], Decimal("10")),),
        date(2024, 1, 1),
        "INITIAL",
    )
    result = EventDrivenBacktest(
        run_id="index-valuation",
        initial_cash=Decimal("100000"),
    ).run(
        (
            _session(date(2024, 1, 1), ("10", "20")),
            _session(signal, ("10", "20")),
            _session(effective, ("20", "20")),
        ),
        IndexReplicationStrategy(
            {date(2024, 1, 1): initial, signal: target},
            cash_buffer_weight=Decimal("0"),
        ),
    )
    replacement_fill = next(fill for fill in result.fills if fill.symbol == SYMBOLS[1])
    assert replacement_fill.quantity == 10000
