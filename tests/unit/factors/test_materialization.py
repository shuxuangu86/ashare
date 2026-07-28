import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from aquant.domain.data_release import DataReleaseId
from aquant.factors.atomic import AtomicFactor, FactorPanelInput, baseline_factor_library
from aquant.factors.materialization import FactorMaterializationEngine, MaterializationRequest
from aquant.factors.materialization.incremental import missing_dates
from aquant.factors.materialization.planner import plan_dependencies


@dataclass
class Loader:
    panel: FactorPanelInput
    calls: int = 0

    def load(self, **_kwargs: object) -> FactorPanelInput:
        self.calls += 1
        return self.panel


def input_panel() -> FactorPanelInput:
    start = date(2026, 1, 30)
    dates = tuple(start + timedelta(days=value) for value in range(5))
    codes = ("000001.SZ", "600000.SH")
    close = np.arange(10.0, 20.0).reshape(5, 2)
    return FactorPanelInput(
        dates,
        codes,
        {
            "close": close,
            "total_market_cap": close * 1_000_000,
            "float_market_cap": close * 700_000,
        },
    )


def request(loader: Loader, factors: tuple[AtomicFactor, ...]) -> MaterializationRequest:
    return MaterializationRequest(
        data_release_id=DataReleaseId("cn_equity_20260728_001"),
        factors=factors,
        start_date=loader.panel.trade_dates[0],
        end_date=loader.panel.trade_dates[-1],
        as_of_time=datetime(2026, 7, 28, tzinfo=UTC),
        universe="all_a_share",
        input_loader=loader,
        calendar=object(),
        config={"mode": "test"},
        code_version="test-code",
        config_hash=hashlib.sha256(b"test-config").hexdigest(),
    )


def test_materialization_is_partitioned_atomic_manifested_and_idempotent(
    tmp_path: Path,
) -> None:
    selected = tuple(
        factor
        for factor in baseline_factor_library()
        if factor.spec.factor_id in {"log_total_market_cap", "log_float_market_cap"}
    )
    loader = Loader(input_panel())
    engine = FactorMaterializationEngine(tmp_path)
    first = engine.materialize(request(loader, selected))
    target = tmp_path / f"materialization={first.materialization_id}"
    assert first.total_rows == 20
    assert len(first.files) == 4
    assert (target / "manifest.json").exists()
    assert all((target / item.path).exists() for item in first.files)
    assert all("factor_family=size" in item.path for item in first.files)
    rows = sum(pq.read_table(target / item.path).num_rows for item in first.files)
    assert rows == first.total_rows

    second = engine.materialize(request(loader, selected))
    assert second == first
    assert loader.calls == 1
    assert not tuple(tmp_path.glob(".factor-tmp-*"))


def test_failed_materialization_never_publishes_partial_output(tmp_path: Path) -> None:
    original = baseline_factor_library()[0]
    broken = AtomicFactor(original.spec, lambda _panel: np.ones((1, 1)))
    loader = Loader(input_panel())
    materialization_request = request(loader, (broken,))
    with pytest.raises(ValueError, match="returned"):
        FactorMaterializationEngine(tmp_path).materialize(materialization_request)
    assert not (tmp_path / f"materialization={materialization_request.materialization_id}").exists()
    assert not tuple(tmp_path.glob(".factor-tmp-*"))


def test_dependency_planner_orders_parents_and_rejects_missing_or_cycles() -> None:
    first, second = baseline_factor_library()[:2]
    child = AtomicFactor(
        second.spec.model_copy(update={"parent_factor_ids": (first.spec.factor_id,)}),
        second.calculator,
    )
    assert plan_dependencies((child, first)) == (first, child)
    missing = AtomicFactor(
        second.spec.model_copy(update={"parent_factor_ids": ("unknown",)}),
        second.calculator,
    )
    with pytest.raises(ValueError, match="missing"):
        plan_dependencies((missing,))
    cycle_first = AtomicFactor(
        first.spec.model_copy(update={"parent_factor_ids": (second.spec.factor_id,)}),
        first.calculator,
    )
    cycle_second = AtomicFactor(
        second.spec.model_copy(update={"parent_factor_ids": (first.spec.factor_id,)}),
        second.calculator,
    )
    with pytest.raises(ValueError, match="cycle"):
        plan_dependencies((cycle_first, cycle_second))


def test_incremental_planner_returns_only_unmaterialized_dates() -> None:
    dates = tuple(date(2026, 1, day) for day in range(1, 6))
    assert missing_dates(dates, (dates[0], dates[2])) == (dates[1], dates[3], dates[4])
    with pytest.raises(ValueError, match="sorted"):
        missing_dates(tuple(reversed(dates)), ())
