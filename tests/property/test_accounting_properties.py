from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

from hypothesis import given
from hypothesis import strategies as st

from aquant.backtest.accounting import PortfolioLedger
from aquant.backtest.matching import Fill
from aquant.domain.enums import Side
from aquant.domain.identifiers import Symbol

SYMBOL = Symbol.parse("600000.XSHG")
START = datetime(2026, 7, 16, 1, 30, tzinfo=UTC)


@given(
    buy_quantity=st.integers(min_value=1, max_value=10_000),
    sell_fraction=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    buy_price=st.integers(min_value=1, max_value=1_000),
    sell_price=st.integers(min_value=1, max_value=1_000),
    buy_fee=st.integers(min_value=0, max_value=100),
    sell_fee=st.integers(min_value=0, max_value=100),
)
def test_cash_and_position_are_exactly_explained_by_fills(
    buy_quantity: int,
    sell_fraction: float,
    buy_price: int,
    sell_price: int,
    buy_fee: int,
    sell_fee: int,
) -> None:
    sell_quantity = min(buy_quantity, int(buy_quantity * sell_fraction))
    initial_cash = Decimal(buy_quantity * buy_price + buy_fee + 1_000_000)
    buy = Fill(
        UUID(int=1),
        UUID(int=101),
        SYMBOL,
        Side.BUY,
        buy_quantity,
        Decimal(buy_price),
        Decimal(buy_fee),
        START,
    )
    ledger = PortfolioLedger(initial_cash)
    ledger.apply_fill(buy)

    expected_cash = initial_cash - buy.notional - buy.fee
    if sell_quantity:
        sell = Fill(
            UUID(int=2),
            UUID(int=102),
            SYMBOL,
            Side.SELL,
            sell_quantity,
            Decimal(sell_price),
            Decimal(sell_fee),
            START + timedelta(days=1),
        )
        ledger.apply_fill(sell)
        expected_cash += sell.notional - sell.fee

    assert ledger.cash == expected_cash
    assert ledger.position(SYMBOL).quantity == buy_quantity - sell_quantity
    assert PortfolioLedger.rebuild(initial_cash, ledger.fills).state_hash == ledger.state_hash
