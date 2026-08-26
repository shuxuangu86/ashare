import hashlib
from datetime import date, timedelta

import numpy as np

from aquant.factors.atomic import FactorPanelInput, baseline_factor_library
from aquant.factors.performance import benchmark_factor_workload


def test_benchmark_records_resources_throughput_and_exact_dag_cache() -> None:
    dates = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(30))
    codes = tuple(f"{index:06d}.SZ" for index in range(10))
    close = np.arange(300, dtype=float).reshape(30, 10) + 10
    panel = FactorPanelInput(
        dates,
        codes,
        {
            "close": close,
            "total_market_cap": close * 1e8,
        },
    )
    by_id = {factor.spec.factor_id: factor for factor in baseline_factor_library()}
    factors = (
        by_id["momentum_5d"],
        by_id["log_total_market_cap"],
        by_id["momentum_5d"],
    )
    result = benchmark_factor_workload(
        scenario="unit",
        panel=panel,
        factors=factors,
        data_release_id="release",
        code_version="test",
        config_hash=hashlib.sha256(b"config").hexdigest(),
    )
    assert result.factor_workloads == 3
    assert result.unique_computations == 2
    assert result.cache_hits == 1
    assert result.cache_hit_rate == 1 / 3
    assert result.factor_rows_per_second > 0
    assert result.panel_bytes == close.nbytes * 2
    assert len(result.content_hash) == 64
