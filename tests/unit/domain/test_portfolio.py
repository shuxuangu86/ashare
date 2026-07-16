from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from aquant.domain.enums import Exchange
from aquant.domain.identifiers import Symbol
from aquant.domain.portfolio import TargetPortfolio, TargetPosition


def _position(code: str, exchange: Exchange, weight: str) -> TargetPosition:
    return TargetPosition(Symbol(code, exchange), Decimal(weight))


def _portfolio(*positions: TargetPosition, **overrides: object) -> TargetPortfolio:
    values: dict[str, object] = {
        "strategy_id": "baseline-v1",
        "trade_date": date(2026, 7, 17),
        "asof_time": datetime(2026, 7, 16, 7, tzinfo=UTC),
        "data_release_id": "cn_equity_20260716_001",
        "signal_version": "signal-v1",
        "positions": positions,
    }
    values.update(overrides)
    return TargetPortfolio(**values)  # type: ignore[arg-type]


def test_target_portfolio_reports_cash_weight() -> None:
    portfolio = _portfolio(
        _position("600000", Exchange.XSHG, "0.4"),
        _position("000001", Exchange.XSHE, "0.35"),
    )

    assert portfolio.cash_weight == Decimal("0.25")


@pytest.mark.parametrize("weight", ["-0.01", "1.01"])
def test_target_weight_must_be_bounded(weight: str) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _position("600000", Exchange.XSHG, weight)


def test_target_portfolio_rejects_duplicate_symbols() -> None:
    position = _position("600000", Exchange.XSHG, "0.2")
    with pytest.raises(ValueError, match="duplicate"):
        _portfolio(position, position)


def test_target_portfolio_rejects_leverage() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        _portfolio(
            _position("600000", Exchange.XSHG, "0.6"),
            _position("000001", Exchange.XSHE, "0.5"),
        )


@pytest.mark.parametrize("field", ["strategy_id", "data_release_id", "signal_version"])
def test_target_portfolio_requires_lineage(field: str) -> None:
    with pytest.raises(ValueError):
        _portfolio(**{field: " "})


def test_target_portfolio_requires_aware_asof_time() -> None:
    with pytest.raises(ValueError, match="timezone"):
        _portfolio(asof_time=datetime(2026, 7, 16, 7))
