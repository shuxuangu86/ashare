import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aquant.factors.evaluation import (
    long_short_return_series,
    pit_forward_return_labels,
)
from aquant.factors.evaluation.ic import information_coefficient
from aquant.factors.selection.cache import ConvergenceCache
from aquant.factors.selection.convergence import converge_factors


def converge_cached_evaluation(
    *,
    cache_root: Path,
    report_dir: Path,
    output: Path,
    horizon: int = 5,
    maximum_distance: float = 0.5,
    selection_fraction: float = 0.8,
) -> dict[str, object]:
    cache = ConvergenceCache(cache_root, verify_content=True)
    metadata = json.loads(cache.metadata_path.read_text(encoding="utf-8"))
    if metadata["status"] != "PASS" or not metadata.get("content_hash"):
        raise ValueError("convergence cache must be finalized before selection")
    if not 0.5 <= selection_fraction < 1:
        raise ValueError("selection fraction must be in [0.5, 1)")
    selection_end = int(len(cache.trade_dates) * selection_fraction)
    if selection_end <= horizon or selection_end >= len(cache.trade_dates):
        raise ValueError("convergence cache is too short for an isolated selection window")
    selection_dates = cache.trade_dates[:selection_end]
    labels = pit_forward_return_labels(
        cache.close[:selection_end],
        selection_dates,
        horizon,
    )[0]
    factor_values = {
        factor_id: cache.values[index, :selection_end]
        for index, factor_id in enumerate(cache.factor_ids)
    }
    long_short: dict[str, npt.ArrayLike] = {}
    rank_ic: dict[str, npt.ArrayLike] = {}
    scores: dict[str, float] = {}
    report_hashes: dict[str, str] = {}
    for factor_id, values in factor_values.items():
        report_path = _single_report(report_dir, factor_id)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_hashes[factor_id] = str(report["content_hash"])
        long_short[factor_id] = long_short_return_series(values, labels)
        ranked = information_coefficient(values, labels, rank=True)
        rank_values = np.asarray(ranked.by_date, dtype=np.float64)
        rank_ic[factor_id] = rank_values
        finite_rank = rank_values[np.isfinite(rank_values)]
        rank_icir = (
            float(np.mean(finite_rank) / np.std(finite_rank))
            if len(finite_rank) > 1 and np.std(finite_rank) > 0
            else 0.0
        )
        coverage = float(np.count_nonzero(np.isfinite(values)) / values.size)
        scores[factor_id] = abs(float(ranked.mean)) * min(abs(rank_icir), 3) * coverage
    result = converge_factors(
        factor_values,
        labels,
        long_short,
        rank_ic,
        scores,
        maximum_distance=maximum_distance,
    )
    representatives = tuple(
        sorted(
            (item.factor_id for item in result.factors if item.is_representative),
            key=lambda factor_id: (-scores[factor_id], factor_id),
        )
    )
    selection_status = "PASS" if 15 <= len(representatives) <= 25 else "REVIEW_REQUIRED"
    payload: dict[str, object] = {
        "status": selection_status,
        "experiment_id": f"five_year_convergence_h{horizon}_v1",
        "data_release_id": cache.data_release_id,
        "cache_content_hash": metadata["content_hash"],
        "report_content_hashes": report_hashes,
        "horizon": horizon,
        "maximum_distance": maximum_distance,
        "selection_fraction": selection_fraction,
        "selection_window": {
            "start_date": selection_dates[0].isoformat(),
            "end_date": selection_dates[-1].isoformat(),
            "purpose": "feature_selection_only",
        },
        "holdout_window": {
            "start_date": cache.trade_dates[selection_end].isoformat(),
            "end_date": cache.trade_dates[-1].isoformat(),
            "used_for_selection": False,
        },
        "score_definition": ("isolated-selection abs(RankIC)*min(abs(RankICIR),3)*coverage"),
        "factor_ids": result.factor_ids,
        "value_spearman": result.value_spearman.tolist(),
        "long_short_correlation": result.long_short_correlation.tolist(),
        "rank_ic_correlation": result.rank_ic_correlation.tolist(),
        "combined_correlation": result.combined_correlation.tolist(),
        "factors": [asdict(item) for item in result.factors],
        "compact_factor_ids": representatives,
        "compact_factor_count": len(representatives),
        "production_core_factor_ids": [],
        "production_core_status": "NOT_ADMITTED",
    }
    payload["content_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json_atomic(output, payload)
    return payload


def _single_report(report_dir: Path, factor_id: str) -> Path:
    matches = tuple(report_dir.glob(f"{factor_id}-*.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one audited report for {factor_id}, found {len(matches)}")
    return matches[0]


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".convergence-result-",
        suffix=".json",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
