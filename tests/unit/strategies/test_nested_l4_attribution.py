from datetime import date
from decimal import Decimal

import pytest
from scripts.analyze_nested_l4_attribution import (
    _boundary_rate,
    _conditional_paths,
    _daily_costs,
)


def test_fill_cost_attribution_reconciles_fee_schedule_and_slippage() -> None:
    fills = [
        {
            "occurred_at": "2024-01-02T01:30:00+00:00",
            "side": "BUY",
            "notional": "10000",
            "fee": "5.10000",
        },
        {
            "occurred_at": "2024-01-03T01:30:00+00:00",
            "side": "SELL",
            "notional": "10000",
            "fee": "10.10000",
        },
    ]
    daily, totals = _daily_costs(fills)

    assert totals["commission"] == Decimal("10")
    assert totals["stamp_duty"] == Decimal("5")
    assert totals["transfer_fee"] == Decimal("0.2")
    assert daily["2024-01-02"]["slippage"] == pytest.approx(Decimal("4.9975012493753123438"))
    assert daily["2024-01-03"]["slippage"] == pytest.approx(Decimal("5.0025012506253126563"))


def test_conditional_paths_add_back_daily_cost_without_lookahead() -> None:
    curve = [
        {
            "trade_date": "2024-01-02",
            "outer_fold": "0",
            "strategy_return": "0.01",
            "benchmark_return": "0",
            "strategy_net_value": "1.01",
        },
        {
            "trade_date": "2024-01-03",
            "outer_fold": "0",
            "strategy_return": "-0.01",
            "benchmark_return": "0",
            "strategy_net_value": "0.9999",
        },
    ]
    costs = {
        "2024-01-02": {"commission": Decimal("100")},
        "2024-01-03": {"commission": Decimal("101")},
    }
    rows, summary = _conditional_paths(curve, costs, Decimal("100000"))

    assert rows[0]["conditional_gross_return"] == pytest.approx(0.011)
    assert rows[1]["conditional_gross_return"] == pytest.approx(-0.009)
    assert summary["conditional_gross_total_return"] == pytest.approx(0.001901)


def test_boundary_liquidation_rate_uses_date_effective_a_share_costs() -> None:
    assert _boundary_rate(date(2023, 7, 1)) == Decimal("0.00181")
    assert _boundary_rate(date(2024, 7, 1)) == Decimal("0.00131")
