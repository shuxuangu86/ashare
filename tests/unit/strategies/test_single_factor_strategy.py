from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from aquant.backtest import EventDrivenBacktest, MarketSession
from aquant.domain.enums import Exchange
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar, LimitStatus, SecurityStatus
from aquant.strategies import (
    SelectionTail,
    SingleFactorConfig,
    SingleFactorEqualWeightStrategy,
    SingleFactorObservation,
    SingleFactorSelector,
    SingleFactorSnapshot,
)

SYMBOLS = (Symbol.parse("600001.XSHG"), Symbol.parse("000002.XSHE"))


def _session(day: int) -> MarketSession:
    trade_date = date(2026, 7, day)
    bars = tuple(
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
        for symbol, price in zip(SYMBOLS, ("10", "20"), strict=True)
    )
    statuses = tuple(
        SecurityStatus(symbol, trade_date, False, False, LimitStatus.NONE) for symbol in SYMBOLS
    )
    return MarketSession(
        trade_date,
        datetime(2026, 7, day, 1, 30, tzinfo=UTC),
        datetime(2026, 7, day, 11, 30, tzinfo=UTC),
        bars,
        statuses,
    )


def _snapshot(day: int, values: tuple[float, float]) -> SingleFactorSnapshot:
    trade_date = date(2026, 7, day)
    asof = datetime(2026, 7, day, 11, 30, tzinfo=UTC)
    return SingleFactorSnapshot(
        trade_date,
        asof,
        tuple(
            SingleFactorObservation(
                symbol=symbol,
                trade_date=trade_date,
                available_at=asof,
                list_date=date(2020, 1, 1),
                factor_value=value,
                suspended=False,
                is_st=False,
                is_delisting_risk=False,
            )
            for symbol, value in zip(SYMBOLS, values, strict=True)
        ),
    )


def test_negative_direction_selects_low_value_and_fills_t_plus_one() -> None:
    signal_date = date(2026, 7, 16)
    config = SingleFactorConfig(
        "amount_concentration_20d",
        "1.0.0",
        -1,
        target_count=1,
        minimum_constituents=1,
    )
    snapshots = {
        signal_date: _snapshot(16, (0.1, 0.2)),
        date(2026, 7, 17): _snapshot(17, (0.1, 0.2)),
    }
    strategy = SingleFactorEqualWeightStrategy(
        snapshots=snapshots,
        rebalance_dates={signal_date},
        config=config,
    )
    result = EventDrivenBacktest(
        run_id="single-factor-test",
        initial_cash=Decimal("100000"),
    ).run((_session(16), _session(17)), strategy)

    assert SingleFactorSelector(config).select(snapshots[signal_date]).symbols == (SYMBOLS[0],)
    assert len(result.fills) == 1
    assert result.fills[0].symbol == SYMBOLS[0]
    assert result.fills[0].occurred_at == _session(17).open_at


def test_fractional_best_and_worst_tails_are_direction_aware() -> None:
    trade_date = date(2026, 7, 16)
    asof = datetime(2026, 7, 16, 11, 30, tzinfo=UTC)
    observations = tuple(
        SingleFactorObservation(
            Symbol(f"{index:06d}", Exchange.XSHE),
            trade_date,
            asof,
            date(2020, 1, 1),
            float(index),
            False,
            False,
            False,
        )
        for index in range(1, 11)
    )
    snapshot = SingleFactorSnapshot(trade_date, asof, observations)
    base = {
        "factor_id": "amount_concentration_20d",
        "factor_version": "1.0.0",
        "expected_direction": -1,
        "target_fraction": Decimal("0.2"),
        "minimum_constituents": 10,
    }
    best = SingleFactorSelector(SingleFactorConfig(**base)).select(snapshot)
    worst = SingleFactorSelector(
        SingleFactorConfig(**base, selection_tail=SelectionTail.WORST)
    ).select(snapshot)
    assert best.symbols == tuple(item.symbol for item in observations[:2])
    assert worst.symbols == tuple(item.symbol for item in reversed(observations[-2:]))


def test_selector_excludes_missing_and_nontradable_observations() -> None:
    snapshot = _snapshot(16, (0.1, 0.2))
    blocked = SingleFactorObservation(
        symbol=snapshot.observations[0].symbol,
        trade_date=snapshot.trade_date,
        available_at=snapshot.asof_time,
        list_date=date(2020, 1, 1),
        factor_value=None,
        suspended=False,
        is_st=False,
        is_delisting_risk=False,
    )
    revised = SingleFactorSnapshot(
        snapshot.trade_date,
        snapshot.asof_time,
        (blocked, snapshot.observations[1]),
    )
    selection = SingleFactorSelector(
        SingleFactorConfig(
            "factor",
            "1.0.0",
            -1,
            target_count=1,
            minimum_constituents=1,
        )
    ).select(revised)
    assert selection.symbols == (SYMBOLS[1],)
    assert dict(selection.excluded_counts)["missing_factor"] == 1


def test_selector_accounts_for_each_explicit_universe_exclusion() -> None:
    trade_date = date(2026, 7, 16)
    asof = datetime(2026, 7, 16, 11, 30, tzinfo=UTC)

    def observation(
        code: str,
        *,
        exchange: Exchange = Exchange.XSHE,
        value: float | None = 0.1,
        list_date: date = date(2020, 1, 1),
        suspended: bool = False,
        is_st: bool = False,
        delisting: bool = False,
    ) -> SingleFactorObservation:
        return SingleFactorObservation(
            Symbol(code, exchange),
            trade_date,
            asof,
            list_date,
            value,
            suspended,
            is_st,
            delisting,
        )

    snapshot = SingleFactorSnapshot(
        trade_date,
        asof,
        (
            observation("000001"),
            observation("000002", exchange=Exchange.XBSE),
            observation("000003", value=None),
            observation("000004", suspended=True),
            observation("000005", is_st=True),
            observation("000006", delisting=True),
            observation("000007", list_date=date(2026, 7, 1)),
        ),
    )
    selection = SingleFactorSelector(
        SingleFactorConfig("factor", "1.0.0", -1, target_count=1, minimum_constituents=1)
    ).select(snapshot)
    assert selection.symbols == (Symbol("000001", Exchange.XSHE),)
    assert dict(selection.excluded_counts) == {
        "delisting_risk": 1,
        "exchange": 1,
        "missing_factor": 1,
        "new_listing": 1,
        "st": 1,
        "suspended": 1,
    }


def test_non_rebalance_st_position_is_exited_at_next_open() -> None:
    config = SingleFactorConfig(
        "amount_concentration_20d",
        "1.0.0",
        -1,
        target_count=1,
        minimum_constituents=1,
    )
    first = _snapshot(15, (0.1, 0.2))
    second_base = _snapshot(16, (0.1, 0.2))
    second = SingleFactorSnapshot(
        second_base.trade_date,
        second_base.asof_time,
        (
            replace(second_base.observations[0], is_st=True),
            second_base.observations[1],
        ),
    )
    result = EventDrivenBacktest(
        run_id="single-factor-hard-exit",
        initial_cash=Decimal("100000"),
    ).run(
        (_session(15), _session(16), _session(17)),
        SingleFactorEqualWeightStrategy(
            snapshots={
                date(2026, 7, 15): first,
                date(2026, 7, 16): second,
                date(2026, 7, 17): _snapshot(17, (0.1, 0.2)),
            },
            rebalance_dates={date(2026, 7, 15)},
            config=config,
        ),
    )
    assert [fill.side.value for fill in result.fills] == ["BUY", "SELL"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"factor_id": ""},
        {"expected_direction": 0},
        {"target_count": 0},
        {"target_fraction": Decimal("1")},
        {"minimum_listing_days": -1},
        {"lot_size": 0},
        {"cash_buffer_weight": Decimal("1")},
    ],
)
def test_single_factor_config_rejects_unsafe_parameters(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "factor_id": "factor",
        "factor_version": "1.0.0",
        "expected_direction": -1,
    }
    with pytest.raises(ValueError):
        SingleFactorConfig(**{**values, **kwargs})  # type: ignore[arg-type]


def test_single_factor_observation_and_snapshot_reject_future_data() -> None:
    trade_date = date(2026, 7, 16)
    asof = datetime(2026, 7, 16, 11, 30, tzinfo=UTC)
    with pytest.raises(ValueError, match="dates"):
        SingleFactorObservation(
            SYMBOLS[0],
            trade_date,
            asof,
            date(2026, 7, 17),
            0.1,
            False,
            False,
            False,
        )
    with pytest.raises(ValueError, match="finite"):
        replace(_snapshot(16, (0.1, 0.2)).observations[0], factor_value=float("nan"))
    observation = replace(
        _snapshot(16, (0.1, 0.2)).observations[0],
        available_at=datetime(2026, 7, 16, 12, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="unavailable"):
        SingleFactorSnapshot(trade_date, asof, (observation,))
