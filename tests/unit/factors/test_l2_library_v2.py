import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from aquant.factors.selection.cache import ConvergenceCache
from aquant.factors.selection.library_v2 import (
    publish_feature_eligible_pool,
    run_l3_family_smoke,
)


def test_publish_pool_deduplicates_and_keeps_family_representative(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "data_release_id": "test-release",
                "code_version": "test-code",
                "config_hash": "a" * 64,
            }
        )
    )
    rows = []
    for factor_id, expression_hash, scale in (
        ("factor_a", "a" * 64, 1.0),
        ("factor_b", "a" * 64, 0.9),
        ("factor_bad", "b" * 64, 0.0),
    ):
        rows.append({"factor_id": factor_id})
        payload = _report(factor_id, expression_hash, scale)
        (reports / f"{factor_id}-1.0.0.json").write_text(json.dumps(payload))
    catalog = tmp_path / "catalog.parquet"
    pl.DataFrame(rows).write_parquet(catalog)

    payload = publish_feature_eligible_pool(
        report_dir=reports,
        catalog_path=catalog,
        output_dir=tmp_path / "output",
    )

    assert payload["factor_count"] == 3
    assert payload["research_validated_count"] == 2
    assert payload["feature_eligible_count"] == 1
    assert payload["exact_duplicate_count"] == 1
    assert payload["rejected_count"] == 1
    assert (tmp_path / "output" / "l2_oos_evaluation_complete.csv").is_file()


def test_l3_smoke_uses_time_ordered_train_only_folds(tmp_path: Path) -> None:
    dates = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(30))
    codes = tuple(f"{index:06d}.SZ" for index in range(8))
    generator = np.random.default_rng(3)
    close = 10 + np.cumsum(generator.normal(0, 0.1, (30, 8)), axis=0)
    cache = ConvergenceCache.create(
        tmp_path / "cache",
        factor_ids=("factor_a", "factor_b"),
        trade_dates=dates,
        ts_codes=codes,
        close=close,
        data_release_id="test-release",
        config_hash="a" * 64,
    )
    cache.write("factor_a", generator.normal(size=(30, 8)))
    cache.write("factor_b", generator.normal(size=(30, 8)))
    cache.finalize()
    pool = {
        "content_hash": "b" * 64,
        "evaluation_code_version": "test-code",
        "factors": [
            {
                "factor_id": factor_id,
                "family": "momentum",
                "direction": direction,
            }
            for factor_id, direction in (("factor_a", 1), ("factor_b", -1))
        ],
    }
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(pool))
    reports = tmp_path / "l3-reports"
    reports.mkdir()
    for factor_id, direction in (("factor_a", 1), ("factor_b", -1)):
        (reports / f"{factor_id}-1.0.0.json").write_text(
            json.dumps({"rank_ic": {"by_date": [0.01 * direction] * 30}})
        )

    payload = run_l3_family_smoke(
        cache_root=tmp_path / "cache",
        report_dir=reports,
        pool_path=pool_path,
        output=tmp_path / "smoke.json",
        minimum_train_dates=12,
    )

    assert payload["status"] == "PASS"
    assert payload["research_status"] == "RESEARCH_ONLY"
    assert payload["train_only_fit"] is True
    assert len(payload["results"]) == 5
    assert all(len(item["fold_rank_ic"]) == 3 for item in payload["results"])


def _report(factor_id: str, expression_hash: str, scale: float) -> dict[str, object]:
    by_date = [scale * value for value in (0.01, 0.02, -0.01, 0.03, 0.02) * 10]
    payload: dict[str, object] = {
        "content_hash": hashlib.sha256(factor_id.encode()).hexdigest(),
        "factor": {
            "factor_id": factor_id,
            "version": "1.0.0",
            "family": "momentum",
            "subfamily": "test",
            "role": "ALPHA_CANDIDATE",
            "source_id": "SRC_TEST",
            "expression_hash": expression_hash,
            "expected_direction": 1,
            "complexity_score": 1.0,
        },
        "rank_ic": {"by_date": by_date},
        "quality": {
            "coverage": 0.9,
            "std": scale,
            "unique_count": 10 if scale else 1,
        },
        "extra_metrics": {
            "oos_horizons": {
                "5": {
                    "rank_ic_mean": 0.02 * scale,
                    "rank_icir": 0.2 * scale,
                    "monotonicity": 0.5 * scale,
                    "turnover": 0.2,
                    "long_short_return": 0.001 * scale,
                    "net_long_short_return": 0.0008 * scale,
                    "annual_rank_ic": [[2025, 0.02 * scale]],
                    "regime_rank_ic": [["BULL", 0.02 * scale]],
                    "newey_west_t": 2.0 * scale,
                }
            }
        },
    }
    return payload
