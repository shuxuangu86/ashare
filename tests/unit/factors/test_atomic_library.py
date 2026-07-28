from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from aquant.domain.data_release import DataReleaseId
from aquant.factors.atomic import FactorPanelInput, baseline_factor_library
from aquant.factors.types import FactorContext

SUPPORTED_STANDARD_PIT_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "total_market_cap",
    "float_market_cap",
    "pb",
    "turnover_rate",
    "dividend_yield",
    "netprofit_yoy",
    "debt_to_assets",
    "up_limit",
    "down_limit",
}


def panel(*, future_multiplier: float = 1.0, cutoff: int = 270) -> FactorPanelInput:
    rows, columns = 300, 5
    timeline = np.arange(rows, dtype=float)[:, None]
    securities = np.arange(columns, dtype=float)[None, :]
    close = 10 + timeline * 0.03 + securities * 0.5 + np.sin(timeline / 7)
    volume = 1_000_000 + timeline * 1000 + securities * 10_000
    fields = {
        "close": close,
        "open": close * (1 + 0.001 * np.cos(timeline)),
        "high": close * 1.02,
        "low": close * 0.98,
        "volume": volume,
        "amount": volume * close,
        "total_market_cap": close * (1_000_000_000 + securities * 10_000_000),
        "float_market_cap": close * (700_000_000 + securities * 8_000_000),
        "pb": 1.2 + securities * 0.1 + timeline * 0.0001,
        "turnover_rate": 1 + 0.1 * np.sin(timeline / 9) + securities * 0.02,
        "dividend_yield": 0.5 + securities * 0.05 + timeline * 0.0002,
        "netprofit_yoy": 5 + securities + 2 * np.sin(timeline / 40),
        "debt_to_assets": 30 + securities + np.cos(timeline / 50),
        "up_limit": close * 1.1,
        "down_limit": close * 0.9,
    }
    if future_multiplier != 1:
        for values in fields.values():
            values[cutoff:] *= future_multiplier
    start = date(2025, 1, 1)
    return FactorPanelInput(
        tuple(start + timedelta(days=index) for index in range(rows)),
        tuple(f"{index:06d}.SZ" for index in range(columns)),
        fields,
    )


def test_baseline_library_has_at_least_fifty_real_supported_factors() -> None:
    factors = baseline_factor_library()
    assert len(factors) == 73
    assert len({factor.spec.factor_id for factor in factors}) == len(factors)
    assert len({factor.spec.family for factor in factors}) >= 8
    assert all(set(factor.spec.input_fields) <= SUPPORTED_STANDARD_PIT_FIELDS for factor in factors)


@pytest.mark.parametrize("factor", baseline_factor_library(), ids=lambda x: x.spec.factor_id)
def test_every_baseline_factor_computes_deterministically(factor: object) -> None:
    resolved_factor = factor
    first = resolved_factor.compute_array(panel())  # type: ignore[attr-defined]
    second = resolved_factor.compute_array(panel())  # type: ignore[attr-defined]
    assert first.shape == (300, 5)
    assert np.count_nonzero(np.isfinite(first)) > 0
    np.testing.assert_allclose(first, second, equal_nan=True)


def test_future_input_changes_do_not_change_past_factor_values() -> None:
    cutoff = 270
    original = panel(cutoff=cutoff)
    changed = panel(future_multiplier=1000, cutoff=cutoff)
    for factor in baseline_factor_library():
        np.testing.assert_allclose(
            factor.compute_array(original)[:cutoff],
            factor.compute_array(changed)[:cutoff],
            equal_nan=True,
            err_msg=factor.spec.factor_id,
        )


@dataclass
class Loader:
    value: FactorPanelInput
    calls: int = 0

    def load(self, **kwargs: object) -> FactorPanelInput:
        assert kwargs["as_of_time"] == datetime(2026, 7, 28, tzinfo=UTC)
        self.calls += 1
        return self.value


def test_atomic_factor_uses_loader_contract_and_emits_stable_long_table() -> None:
    loaded = panel()
    loader = Loader(loaded)
    context = FactorContext(
        start_date=loaded.trade_dates[0],
        end_date=loaded.trade_dates[-1],
        as_of_time=datetime(2026, 7, 28, tzinfo=UTC),
        universe_id="all_a_share",
        data_release_id=DataReleaseId("cn_equity_20260728_001"),
        input_loader=loader,
        calendar=object(),
        config={},
    )
    factor = baseline_factor_library()[0]
    result = factor.compute(context)
    assert loader.calls == 1
    assert len(result.values) == 1500
    assert result.values[0].trade_date == loaded.trade_dates[0]
    assert result.values[0].ts_code == loaded.ts_codes[0]
    assert result.values[-1].ts_code == loaded.ts_codes[-1]
