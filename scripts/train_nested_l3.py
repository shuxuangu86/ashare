#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from aquant.domain.data_release import DataReleaseId
from aquant.factors.aggregation.nested_l3 import run_nested_l3
from aquant.factors.data_loader import StandardPITFactorLoader
from aquant.factors.selection.cache import ConvergenceCache
from aquant.strategies.all_a_equal_proxy import build_all_a_equal_weight_proxy
from aquant.strategies.nested_l4 import build_l4_eligibility_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Train strict outer-fold L3 family models")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--history-release", type=Path, required=True)
    parser.add_argument("--fold-pools", type=Path, required=True)
    parser.add_argument("--output-scores", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--inner-validation-dates", type=int, default=252)
    parser.add_argument("--inner-purge-dates", type=int, default=6)
    parser.add_argument("--maximum-training-rows", type=int, default=200_000)
    parser.add_argument("--code-version", required=True)
    args = parser.parse_args()
    cache = ConvergenceCache(args.cache_dir, verify_content=True)
    metadata = json.loads(cache.metadata_path.read_text())
    if metadata.get("status") != "PASS" or not metadata.get("content_hash"):
        raise ValueError("nested union convergence cache is incomplete")
    pools = json.loads(args.fold_pools.read_text())
    benchmark_points, benchmark_metadata = build_all_a_equal_weight_proxy(
        args.history_release,
        start_date=cache.trade_dates[0],
        end_date=cache.trade_dates[-1],
    )
    benchmark_by_date = {point.trade_date: point.return_rate for point in benchmark_points}
    benchmark_returns = np.asarray(
        [benchmark_by_date[trade_date] for trade_date in cache.trade_dates], dtype=np.float64
    )
    eligibility = build_l4_eligibility_matrix(
        args.history_release,
        trade_dates=cache.trade_dates,
        ts_codes=cache.ts_codes,
    )
    eligibility_hash = hashlib.sha256(np.ascontiguousarray(eligibility).tobytes()).hexdigest()
    market_panel = StandardPITFactorLoader(args.history_release).load(
        fields=("open",),
        start_date=cache.trade_dates[0],
        end_date=cache.trade_dates[-1],
        as_of_time=datetime.combine(
            cache.trade_dates[-1], datetime.max.time(), ZoneInfo("Asia/Shanghai")
        ),
        universe_id="all_a_share",
        data_release_id=DataReleaseId(cache.data_release_id),
    )
    if market_panel.trade_dates != cache.trade_dates or market_panel.ts_codes != cache.ts_codes:
        raise ValueError("adjusted open panel does not align with convergence cache")
    payload = run_nested_l3(
        factor_values=cache.values,
        close=cache.close,
        trade_dates=cache.trade_dates,
        factor_ids=cache.factor_ids,
        fold_pools=pools,
        output_scores=args.output_scores,
        output_metadata=args.output_metadata,
        horizon=args.horizon,
        inner_validation_dates=args.inner_validation_dates,
        inner_purge_dates=args.inner_purge_dates,
        maximum_training_rows=args.maximum_training_rows,
        neutralization=cache.neutralization,
        benchmark_returns=benchmark_returns,
        eligibility_mask=eligibility,
        open_prices=market_panel.fields["open"],
        provenance={
            "data_release_id": cache.data_release_id,
            "cache_config_hash": cache.config_hash,
            "cache_content_hash": metadata["content_hash"],
            "price_basis": cache.price_basis,
            "neutralization": cache.neutralization,
            "benchmark_points_hash": benchmark_metadata["points_hash"],
            "eligibility_hash": eligibility_hash,
            "code_version": args.code_version,
        },
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "fold_count": len(payload["folds"]),
                "selected_method_counts": payload["selected_method_counts"],
                "content_hash": payload["content_hash"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
