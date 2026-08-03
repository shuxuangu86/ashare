#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aquant.factors.aggregation.nested_l3 import run_nested_l3
from aquant.factors.selection.cache import ConvergenceCache


def main() -> None:
    parser = argparse.ArgumentParser(description="Train strict outer-fold L3 family models")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--fold-pools", type=Path, required=True)
    parser.add_argument("--output-scores", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--inner-validation-dates", type=int, default=252)
    parser.add_argument("--inner-purge-dates", type=int, default=6)
    parser.add_argument("--maximum-training-rows", type=int, default=200_000)
    args = parser.parse_args()
    cache = ConvergenceCache(args.cache_dir)
    metadata = json.loads(cache.metadata_path.read_text())
    if metadata.get("status") != "PASS" or not metadata.get("content_hash"):
        raise ValueError("nested union convergence cache is incomplete")
    pools = json.loads(args.fold_pools.read_text())
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
        provenance={
            "data_release_id": cache.data_release_id,
            "cache_config_hash": cache.config_hash,
            "cache_content_hash": metadata["content_hash"],
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
