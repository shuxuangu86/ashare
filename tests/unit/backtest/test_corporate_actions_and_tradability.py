from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest

from aquant.backtest import (
    CorporateAction,
    CorporateActionKind,
    EventDrivenBacktest,
    MarketSession,
    OrderRequest,
    StrategyContext,
    TradabilityReason,
    TradabilityRules,
    evaluate_tradability,
    tradability_mask,
)
from aquant.backtest.accounting import PortfolioLedger
from aquant.backtest.matching import Fill
from aquant.domain.enums import Exchange, Side
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar, LimitStatus, SecurityStatus

SYMBOL = Symbol.parse("600000.XSHG")
START = datetime(2026, 7, 1, 1, 30, tzinfo=UTC)


def _fill() -> Fill:
    return Fill(
        UUID(int=1),
        UUID(int=101),
        SYMBOL,
        Side.BUY,
        100,
        Decimal("10"),
        Decimal("0"),
        START,
    )


def _action(sequence: int, kind: CorporateActionKind, **updates: object) -> CorporateAction:
    values: dict[str, object] = {
        "action_id": UUID(int=sequence),
        "symbol": SYMBOL,
        "kind": kind,
        "occurred_at": START + timedelta(days=sequence),
    }
    values.update(updates)
    return CorporateAction(**values)  # type: ignore[arg-type]


def test_dividend_stock_rights_and_delisting_preserve_accounting_equation() -> None:
    ledger = PortfolioLedger(Decimal("2000"))
    ledger.apply_fill(_fill())
    before = ledger.snapshot(
        asof_time=START,
        prices={SYMBOL: Decimal("10")},
    )

    entitlement = _action(
        2,
        CorporateActionKind.CASH_DIVIDEND_ENTITLEMENT,
        cash_per_share=Decimal("1"),
    )
    ledger.apply_corporate_action(entitlement)
    ex_dividend = ledger.snapshot(
        asof_time=entitlement.occurred_at,
        prices={SYMBOL: Decimal("9")},
    )
    assert ex_dividend.equity == before.equity
    assert ex_dividend.receivables == Decimal("100")
    assert ledger.position(SYMBOL).average_cost == Decimal("9")

    payment = _action(
        3,
        CorporateActionKind.CASH_DIVIDEND_PAYMENT,
        reference_action_id=entitlement.action_id,
    )
    ledger.apply_corporate_action(payment)
    assert ledger.cash == Decimal("1100")
    assert ledger.receivables == 0

    stock = _action(4, CorporateActionKind.STOCK_DIVIDEND, ratio=Decimal("1"))
    ledger.apply_corporate_action(stock)
    assert ledger.position(SYMBOL).quantity == 200
    assert ledger.position(SYMBOL).average_cost == Decimal("4.5")

    rights = _action(
        5,
        CorporateActionKind.RIGHTS_ISSUE,
        ratio=Decimal("0.5"),
        subscription_price=Decimal("4"),
    )
    ledger.apply_corporate_action(rights)
    after_rights = ledger.snapshot(
        asof_time=rights.occurred_at,
        prices={SYMBOL: Decimal(1300) / Decimal(300)},
    )
    assert after_rights.equity == before.equity

    delisting = _action(
        6,
        CorporateActionKind.DELISTING,
        settlement_price=Decimal(1300) / Decimal(300),
    )
    ledger.apply_corporate_action(delisting)
    assert ledger.position(SYMBOL).quantity == 0
    assert ledger.cash == before.equity
    rebuilt = PortfolioLedger.rebuild(
        Decimal("2000"),
        ledger.fills,
        ledger.corporate_actions,
    )
    assert rebuilt.state_hash == ledger.state_hash


def _bar(**updates: object) -> DailyBar:
    values: dict[str, object] = {
        "symbol": SYMBOL,
        "trade_date": date(2026, 7, 2),
        "open": Decimal("10"),
        "high": Decimal("10.5"),
        "low": Decimal("9.5"),
        "close": Decimal("10"),
        "volume": Decimal("10000"),
        "amount": Decimal("100000"),
    }
    values.update(updates)
    return DailyBar(**values)  # type: ignore[arg-type]


def _status(**updates: object) -> SecurityStatus:
    values: dict[str, object] = {
        "symbol": SYMBOL,
        "trade_date": date(2026, 7, 2),
        "suspended": False,
        "is_st": False,
        "limit_status": LimitStatus.NONE,
        "listing_age_days": 1000,
    }
    values.update(updates)
    return SecurityStatus(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("side", "bar", "status", "rules", "reason"),
    [
        (
            Side.BUY,
            None,
            _status(),
            TradabilityRules(),
            TradabilityReason.MISSING_MARKET_DATA,
        ),
        (
            Side.BUY,
            _bar(),
            None,
            TradabilityRules(),
            TradabilityReason.MISSING_STATUS,
        ),
        (
            Side.BUY,
            _bar(),
            _status(is_listed=False),
            TradabilityRules(),
            TradabilityReason.NOT_LISTED,
        ),
        (
            Side.BUY,
            _bar(),
            _status(is_listed=False, is_delisted=True),
            TradabilityRules(),
            TradabilityReason.DELISTED,
        ),
        (
            Side.BUY,
            _bar(),
            _status(suspended=True),
            TradabilityRules(),
            TradabilityReason.SUSPENDED,
        ),
        (Side.BUY, _bar(), _status(is_st=True), TradabilityRules(), TradabilityReason.ST),
        (
            Side.BUY,
            _bar(high=Decimal("10"), low=Decimal("10")),
            _status(limit_status=LimitStatus.LIMIT_UP),
            TradabilityRules(),
            TradabilityReason.ONE_PRICE_BOARD,
        ),
        (
            Side.SELL,
            _bar(),
            _status(limit_status=LimitStatus.LIMIT_DOWN),
            TradabilityRules(),
            TradabilityReason.LIMIT_DOWN_SELL,
        ),
        (
            Side.BUY,
            _bar(),
            _status(),
            TradabilityRules(minimum_amount=Decimal("200000")),
            TradabilityReason.INSUFFICIENT_AMOUNT,
        ),
        (
            Side.BUY,
            _bar(),
            _status(listing_age_days=10),
            TradabilityRules(minimum_listing_days=60),
            TradabilityReason.INSUFFICIENT_LISTING_AGE,
        ),
        (
            Side.BUY,
            _bar(),
            _status(),
            TradabilityRules(allowed_exchanges=(Exchange.XSHE,)),
            TradabilityReason.OUTSIDE_MARKET_SCOPE,
        ),
    ],
)
def test_tradability_reasons_fail_closed(
    side: Side,
    bar: DailyBar | None,
    status: SecurityStatus | None,
    rules: TradabilityRules,
    reason: TradabilityReason,
) -> None:
    decision = evaluate_tradability(side=side, bar=bar, status=status, rules=rules)
    assert not decision.allowed
    assert reason in decision.reasons


def test_tradability_mask_is_side_specific_and_deterministic() -> None:
    bar = _bar()
    status = _status(limit_status=LimitStatus.LIMIT_UP)
    assert tradability_mask(
        ((bar, status), (bar, _status())),
        side=Side.BUY,
        rules=TradabilityRules(),
    ) == (False, True)
    assert evaluate_tradability(
        side=Side.SELL,
        bar=bar,
        status=status,
        rules=TradabilityRules(),
    ).allowed


def test_actions_reject_duplicates_fractional_shares_and_unfunded_rights() -> None:
    ledger = PortfolioLedger(Decimal("1000"))
    ledger.apply_fill(_fill())
    action = _action(2, CorporateActionKind.SPLIT, ratio=Decimal("1.005"))
    with pytest.raises(ValueError, match="fractional"):
        ledger.apply_corporate_action(action)
    rights = _action(
        3,
        CorporateActionKind.RIGHTS_ISSUE,
        ratio=Decimal("1"),
        subscription_price=Decimal("100"),
    )
    with pytest.raises(ValueError, match="cash"):
        ledger.apply_corporate_action(rights)


def test_stock_dividend_cash_in_lieu_preserves_fractional_value() -> None:
    ledger = PortfolioLedger(Decimal("1000"))
    ledger.apply_fill(_fill())
    action = _action(
        2,
        CorporateActionKind.STOCK_DIVIDEND,
        ratio=Decimal("0.015"),
        cash_in_lieu_price=Decimal("9"),
    )
    ledger.apply_corporate_action(action)
    snapshot = ledger.snapshot(
        asof_time=datetime(2026, 7, 2, 7, tzinfo=UTC),
        prices={SYMBOL: Decimal("9")},
    )
    assert snapshot.positions[0].quantity == 101
    assert snapshot.cash == Decimal("4.500")
    assert snapshot.equity == Decimal("913.500")


class _BuyOnce:
    def on_close(self, context: StrategyContext) -> tuple[OrderRequest, ...]:
        if context.session.trade_date == date(2026, 7, 1):
            return (OrderRequest(SYMBOL, Side.BUY, 100),)
        return ()


def _session(day: int, price: str, actions: tuple[CorporateAction, ...] = ()) -> MarketSession:
    trade_date = date(2026, 7, day)
    value = Decimal(price)
    return MarketSession(
        trade_date,
        datetime(2026, 7, day, 1, 30, tzinfo=UTC),
        datetime(2026, 7, day, 7, tzinfo=UTC),
        (
            DailyBar(
                SYMBOL,
                trade_date,
                value,
                value,
                value,
                value,
                Decimal("10000"),
                value * 10000,
            ),
        ),
        (_status(trade_date=trade_date),),
        actions,
    )


def test_event_engine_applies_actions_before_open_fills_and_replays() -> None:
    entitlement = CorporateAction(
        UUID(int=991),
        SYMBOL,
        CorporateActionKind.CASH_DIVIDEND_ENTITLEMENT,
        datetime(2026, 7, 3, 1, 30, tzinfo=UTC),
        cash_per_share=Decimal("1"),
    )
    result = EventDrivenBacktest(
        run_id="corporate-action-integration",
        initial_cash=Decimal("2000"),
    ).run(
        (
            _session(1, "10"),
            _session(2, "10"),
            _session(3, "9", (entitlement,)),
        ),
        _BuyOnce(),
    )
    assert result.corporate_actions == (entitlement,)
    assert result.equity_curve[-1].equity == Decimal("2000")
    assert result.equity_curve[-1].receivables == Decimal("100")
