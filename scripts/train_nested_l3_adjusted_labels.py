#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aquant.factors.aggregation.nested_l3 import (  # type: ignore[import-untyped]
    run_nested_l3,
)
from aquant.factors.selection.cache import ConvergenceCache  # type: ignore[import-untyped]


def main() -> None:
    args = _arguments()
    cache = ConvergenceCache(args.cache_dir)
    cache_metadata = json.loads(cache.metadata_path.read_text())
    matrix_metadata = json.loads((args.market_matrices / "market_matrices.json").read_text())
    if cache_metadata.get("status") != "PASS" or matrix_metadata.get("status") != "PASS":
        raise ValueError("adjusted-label nested L3 inputs are incomplete")
    if matrix_metadata["cache_content_hash"] != cache_metadata["content_hash"]:
        raise ValueError("adjusted market matrices and convergence cache conflict")
    adjusted_close = np.load(args.market_matrices / "adjusted_close.npy", mmap_mode="r")
    if adjusted_close.shape != cache.close.shape:
        raise ValueError("adjusted close does not align with convergence cache")
    pools = json.loads(args.fold_pools.read_text())
    payload = run_nested_l3(
        factor_values=cache.values,
        close=adjusted_close,
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
            "cache_content_hash": cache_metadata["content_hash"],
            "market_matrix_content_hash": matrix_metadata["content_hash"],
            "label_price_basis": "ADJUSTED_CLOSE",
            "diagnostic_retraining_after_outer_oos_observed": True,
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
            }
        )
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain nested L3 with adjusted-close labels")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--market-matrices", type=Path, required=True)
    parser.add_argument("--fold-pools", type=Path, required=True)
    parser.add_argument("--output-scores", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--inner-validation-dates", type=int, default=252)
    parser.add_argument("--inner-purge-dates", type=int, default=6)
    parser.add_argument("--maximum-training-rows", type=int, default=200_000)
    parser.add_argument("--code-version", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
