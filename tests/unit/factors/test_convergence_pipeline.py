from datetime import date, timedelta
from pathlib import Path

import numpy as np

from aquant.factors.atomic import baseline_factor_library
from aquant.factors.evaluation.quality import evaluate_quality
from aquant.factors.reporting import FactorReport, write_factor_report
from aquant.factors.selection import ConvergenceCache, converge_cached_evaluation


def test_cached_convergence_emits_audited_three_view_selection(tmp_path: Path) -> None:
    factors = baseline_factor_library()[:3]
    factor_ids = tuple(factor.spec.factor_id for factor in factors)
    dates = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(30))
    codes = tuple(f"{index:06d}.SZ" for index in range(10))
    generator = np.random.default_rng(11)
    close = 10 + np.cumsum(generator.normal(scale=0.1, size=(30, 10)), axis=0)
    cache = ConvergenceCache.create(
        tmp_path / "cache",
        factor_ids=factor_ids,
        trade_dates=dates,
        ts_codes=codes,
        close=close,
        data_release_id="cn_equity_20260717_001",
        config_hash="a" * 64,
    )
    reports = tmp_path / "reports"
    base = generator.normal(size=close.shape)
    for index, factor in enumerate(factors):
        values = base + generator.normal(scale=0.05 * (index + 1), size=base.shape)
        cache.write(factor.spec.factor_id, values)
        write_factor_report(
            FactorReport(
                factor.spec,
                "cn_equity_20260717_001",
                quality=evaluate_quality(values),
                extra_metrics={
                    "horizons": {
                        "5": {
                            "rank_ic_mean": 0.02 + index / 100,
                            "rank_icir": 0.2 + index / 10,
                            "turnover": 0.3,
                        }
                    }
                },
            ),
            reports,
        )
    cache.finalize()
    output = tmp_path / "selection.json"
    payload = converge_cached_evaluation(
        cache_root=tmp_path / "cache",
        report_dir=reports,
        output=output,
        maximum_distance=0.2,
    )
    assert payload["status"] == "REVIEW_REQUIRED"
    assert payload["compact_factor_count"] == 1
    assert payload["production_core_factor_ids"] == []
    assert len(payload["content_hash"]) == 64
    assert output.is_file()
