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
) -> dict[str, object]:
    cache = ConvergenceCache(cache_root)
    metadata = json.loads(cache.metadata_path.read_text(encoding="utf-8"))
    if metadata["status"] != "PASS" or not metadata.get("content_hash"):
        raise ValueError("convergence cache must be finalized before selection")
    labels = pit_forward_return_labels(cache.close, cache.trade_dates, horizon)[0]
    factor_values = {
        factor_id: cache.values[index] for index, factor_id in enumerate(cache.factor_ids)
    }
    long_short: dict[str, npt.ArrayLike] = {}
    rank_ic: dict[str, npt.ArrayLike] = {}
    scores: dict[str, float] = {}
    report_hashes: dict[str, str] = {}
    for factor_id, values in factor_values.items():
        report_path = _single_report(report_dir, factor_id)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_hashes[factor_id] = str(report["content_hash"])
        metrics = report["extra_metrics"]["horizons"][str(horizon)]
        quality = report["quality"]
        long_short[factor_id] = long_short_return_series(values, labels)
        rank_ic[factor_id] = np.asarray(
            information_coefficient(values, labels, rank=True).by_date,
            dtype=float,
        )
        scores[factor_id] = (
            abs(float(metrics["rank_ic_mean"]))
            * min(abs(float(metrics["rank_icir"])), 3)
            * float(quality["coverage"])
            * max(0.0, 1 - float(metrics["turnover"]))
        )
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
        "score_definition": ("abs(RankIC)*min(abs(RankICIR),3)*coverage*max(0,1-turnover)"),
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
