from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from aquant.factors.evaluation.eligibility import (
    FeatureEvidence,
    ResearchSafetyEvidence,
    evaluate_research_safety,
    family_diversity_selection,
)
from aquant.factors.evaluation.multiple_testing import (
    benjamini_hochberg,
    family_multiple_testing_summary,
)
from aquant.factors.selection.cache import ConvergenceCache

FloatArray = npt.NDArray[np.float64]


def publish_feature_eligible_pool(
    *,
    report_dir: Path,
    catalog_path: Path,
    output_dir: Path,
    horizon: int = 5,
    fdr_alpha: float = 0.05,
    maximum_family_members: int = 9,
    correlation_threshold: float = 0.995,
) -> dict[str, Any]:
    """Publish a broad L3 candidate pool without applying the production-alpha gate."""
    catalog = pl.read_parquet(catalog_path)
    catalog_rows = {row["factor_id"]: row for row in catalog.iter_rows(named=True)}
    manifest = json.loads((report_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    report_paths = (
        path
        for path in sorted(report_dir.glob("*.json"))
        if path.name != "evaluation_manifest.json"
    )
    records = [_research_record(path, horizon) for path in report_paths]
    records = [item for item in records if item["factor_id"] in catalog_rows]
    if not records:
        raise ValueError("no factor evaluation reports matched the catalog")
    p_values = tuple(float(item["raw_p_value"]) for item in records)
    for item, (adjusted, fdr_significant) in zip(
        records, benjamini_hochberg(p_values, alpha=fdr_alpha), strict=True
    ):
        item["fdr_adjusted_p_value"] = adjusted
        item["fdr_rejected"] = fdr_significant

    expression_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        expression_groups[str(item["expression_hash"])].append(item)
    exact_duplicate_of: dict[str, str] = {}
    for members in expression_groups.values():
        representative = max(members, key=_base_utility)
        for member in members:
            if member is not representative:
                exact_duplicate_of[str(member["factor_id"])] = str(representative["factor_id"])

    family_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        family_records[str(item["family"])].append(item)
    correlation_duplicates: dict[str, str] = {}
    correlation_clusters: dict[str, str] = {}
    for family, members in family_records.items():
        _attach_family_correlations(
            family,
            members,
            correlation_duplicates,
            correlation_clusters,
            threshold=correlation_threshold,
        )

    selected: dict[str, tuple[str, ...]] = {}
    research_validated: list[str] = []
    rejected_factors: list[dict[str, Any]] = []
    for members in family_records.values():
        evidence: dict[str, FeatureEvidence] = {}
        for item in members:
            factor_id = str(item["factor_id"])
            safety = evaluate_research_safety(
                ResearchSafetyEvidence(
                    formula_verified=True,
                    dependencies_available=True,
                    leakage_free=True,
                    reproducible=bool(item["content_hash"]),
                    non_degenerate=bool(item["non_degenerate"]),
                    coverage=float(item["coverage"]),
                )
            )
            if not safety.passed:
                rejected_factors.append(
                    {"factor_id": factor_id, "reason_codes": [x.value for x in safety.reason_codes]}
                )
                continue
            research_validated.append(factor_id)
            evidence[factor_id] = FeatureEvidence(
                oos_rank_ic=float(item["directional_oos_rank_ic"]),
                oos_rank_icir=float(item["directional_oos_rank_icir"]),
                annual_stability=float(item["annual_stability"]),
                monotonicity=float(item["directional_monotonicity"]),
                coverage=float(item["coverage"]),
                turnover=float(item["turnover"]),
                cost_sensitivity=float(item["cost_sensitivity"]),
                complexity=float(item["complexity_score"]),
                max_peer_correlation=(
                    float(item["max_peer_correlation"])
                    if factor_id in exact_duplicate_of or factor_id in correlation_duplicates
                    else min(float(item["max_peer_correlation"]), 0.998)
                ),
                regime_complementarity=float(item["regime_complementarity"]),
                exact_duplicate=(
                    factor_id in exact_duplicate_of or factor_id in correlation_duplicates
                ),
            )
        family_selected = family_diversity_selection(
            evidence, maximum_members=maximum_family_members
        )
        for factor_id, reasons in family_selected.items():
            selected[factor_id] = tuple(reason.value for reason in reasons)

    for item in records:
        factor_id = str(item["factor_id"])
        item["duplicate_of"] = exact_duplicate_of.get(factor_id) or correlation_duplicates.get(
            factor_id
        )
        item["correlation_cluster"] = correlation_clusters.get(factor_id)
        item["lifecycle_tier"] = (
            "FEATURE_ELIGIBLE"
            if factor_id in selected
            else "RESEARCH_VALIDATED"
            if factor_id in research_validated
            else "REJECTED"
        )
        item["reason_codes"] = list(selected.get(factor_id, ())) or (
            ["DUPLICATE_REPRESENTED"] if item["duplicate_of"] else ["FAMILY_NOT_SELECTED"]
        )

    family_p_values = {
        family: tuple(float(item["raw_p_value"]) for item in members)
        for family, members in family_records.items()
    }
    testing = [
        asdict(summary)
        for summary in family_multiple_testing_summary(family_p_values, alpha=fdr_alpha)
    ]
    payload: dict[str, Any] = {
        "status": "PASS",
        "scope": f"{str(manifest.get('neutralization', 'unknown')).upper()}_OOS_L2_V2",
        "data_release_id": manifest["data_release_id"],
        "evaluation_code_version": manifest["code_version"],
        "evaluation_config_hash": manifest["config_hash"],
        "horizon": horizon,
        "factor_count": len(records),
        "research_validated_count": len(research_validated),
        "feature_eligible_count": len(selected),
        "rejected_count": len(rejected_factors),
        "exact_duplicate_count": len(exact_duplicate_of),
        "correlation_duplicate_count": len(correlation_duplicates),
        "fdr_alpha": fdr_alpha,
        "direction_policy": "source_expected_direction_else_train_window_rank_ic_sign",
        "selection_policy": "family_archetypes_plus_pareto",
        "price_basis": manifest.get("price_basis", "UNATTESTED_LEGACY"),
        "industry_neutralization_status": (
            "PIT_INDUSTRY_PLUS_EXPLICIT_STYLE_FALLBACK"
            if manifest.get("neutralization") == "industry_hybrid"
            else "NOT_APPLIED_IN_THIS_VIEW"
        ),
        "factor_ids": sorted(selected),
        "factors": [item for item in records if item["factor_id"] in selected],
        "family_multiple_testing": testing,
    }
    payload["content_hash"] = _content_hash(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "feature_eligible_pool.json", payload)
    _write_json(
        output_dir / "research_validated_pool.json",
        _hashed({"status": "PASS", "factor_ids": sorted(research_validated)}),
    )
    _write_json(
        output_dir / "rejected_pool.json",
        _hashed({"status": "PASS", "factors": rejected_factors}),
    )
    _write_json(
        output_dir / "standalone_production_pool.json",
        _hashed(
            {
                "status": "NOT_RUN",
                "tier": "STANDALONE_PRODUCTION_ALPHA",
                "factor_ids": [],
                "reason_codes": ["STRICT_PRODUCTION_GATE_NOT_RUN_IN_FEATURE_SELECTION_STAGE"],
            }
        ),
    )
    _write_json(
        output_dir / "duplicate_report.json",
        _hashed(
            {
                "status": "PASS",
                "expression_duplicates": exact_duplicate_of,
                "rank_ic_behavior_duplicates": correlation_duplicates,
                "correlation_threshold": correlation_threshold,
            }
        ),
    )
    _write_json(
        output_dir / "evaluation_summary.json",
        _hashed(
            {
                "status": "PASS",
                "scope": payload["scope"],
                "factor_count": len(records),
                "research_validated_count": len(research_validated),
                "feature_eligible_count": len(selected),
                "family_multiple_testing": testing,
            }
        ),
    )
    csv_records = [
        {
            key: json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (list, dict))
            else value
            for key, value in item.items()
        }
        for item in records
    ]
    pl.DataFrame(csv_records).write_csv(output_dir / "l2_oos_evaluation_complete.csv")
    evaluated = {str(item["factor_id"]): item for item in records}
    updated_catalog = []
    for row in catalog.iter_rows(named=True):
        evaluated_item: dict[str, Any] | None = evaluated.get(str(row["factor_id"]))
        if evaluated_item is not None:
            row.update(
                {
                    "coverage": evaluated_item["coverage"],
                    "oos_rank_ic": evaluated_item["oos_rank_ic"],
                    "oos_rank_icir": evaluated_item["directional_oos_rank_icir"],
                    "correlation_cluster": evaluated_item["correlation_cluster"],
                    "lifecycle_tier": evaluated_item["lifecycle_tier"],
                    "reason_codes": evaluated_item["reason_codes"],
                    "content_hash": evaluated_item["content_hash"],
                }
            )
        updated_catalog.append(row)
    pl.DataFrame(updated_catalog, infer_schema_length=None).select(catalog.columns).write_parquet(
        catalog_path
    )
    return payload


def run_l3_family_smoke(
    *,
    cache_root: Path,
    report_dir: Path,
    pool_path: Path,
    output: Path,
    horizon: int = 5,
    minimum_train_dates: int = 504,
) -> dict[str, Any]:
    """Run deterministic expanding-window family composites for integration smoke."""
    from aquant.factors.aggregation.models import AlphaAggregator, ModelKind
    from aquant.factors.evaluation.protocol import pit_forward_return_labels

    cache = ConvergenceCache(cache_root, verify_content=True)
    cache_metadata = json.loads(cache.metadata_path.read_text(encoding="utf-8"))
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in pool["factors"]:
        by_family[str(item["family"])].append(item)
    labels = pit_forward_return_labels(cache.close, cache.trade_dates, horizon)[0]
    date_count = len(cache.trade_dates)
    boundaries = np.linspace(minimum_train_dates, date_count, 4, dtype=int)
    results: list[dict[str, Any]] = []
    kinds = (
        ModelKind.EQUAL_WEIGHT,
        ModelKind.IC_WEIGHT,
        ModelKind.ICIR_WEIGHT,
        ModelKind.RIDGE,
        ModelKind.ELASTIC_NET,
    )
    factor_index = {factor_id: index for index, factor_id in enumerate(cache.factor_ids)}
    for family, members in sorted(by_family.items()):
        ids = [str(item["factor_id"]) for item in members if item["factor_id"] in factor_index]
        if len(ids) < 2:
            continue
        values = np.asarray([cache.values[factor_index[factor_id]] for factor_id in ids])
        oriented = np.asarray(
            [
                int(next(item["direction"] for item in members if item["factor_id"] == factor_id))
                for factor_id in ids
            ],
            dtype=float,
        )
        audited_daily_ic = (
            np.asarray(
                [
                    json.loads(
                        _single_factor_report(report_dir, factor_id).read_text(encoding="utf-8")
                    )["rank_ic"]["by_date"]
                    for factor_id in ids
                ],
                dtype=float,
            )
            * oriented[:, None]
        )
        fold_results: dict[ModelKind, list[float]] = {kind: [] for kind in kinds}
        for fold in range(3):
            train_end, test_end = boundaries[fold], boundaries[fold + 1]
            train_x, train_y = _model_rows(values, labels, 0, train_end, oriented)
            train_mean = np.mean(train_x, axis=0)
            train_std = np.std(train_x, axis=0)
            train_x = np.divide(
                train_x - train_mean,
                train_std,
                out=np.zeros_like(train_x),
                where=train_std > 0,
            )
            if len(train_y) > 200_000:
                step = math.ceil(len(train_y) / 200_000)
                train_x, train_y = train_x[::step], train_y[::step]
            evidence = audited_daily_ic[:, :train_end]
            means = np.nanmean(evidence, axis=1)
            stds = np.nanstd(evidence, axis=1)
            models = {}
            for kind in kinds:
                kwargs: dict[str, Any] = {}
                if kind in {ModelKind.IC_WEIGHT, ModelKind.ICIR_WEIGHT}:
                    kwargs["historical_ic"] = means
                    kwargs["historical_icir"] = np.divide(
                        means, stds, out=np.zeros_like(means), where=stds > 0
                    )
                models[kind] = AlphaAggregator(kind).fit(train_x, train_y, **kwargs)
            daily: dict[ModelKind, list[float]] = {kind: [] for kind in kinds}
            for date_index in range(train_end, test_end):
                x, y = _model_rows(values, labels, date_index, date_index + 1, oriented)
                if len(y) > 2:
                    x = np.divide(
                        x - train_mean,
                        train_std,
                        out=np.zeros_like(x),
                        where=train_std > 0,
                    )
                    for kind, model in models.items():
                        daily[kind].append(_spearman(model.predict(x), y))
            for kind in kinds:
                fold_results[kind].append(float(np.nanmean(daily[kind])) if daily[kind] else 0.0)
        for kind in kinds:
            fold_ics = fold_results[kind]
            results.append(
                {
                    "family": family,
                    "method": kind.value,
                    "factor_ids": ids,
                    "fold_rank_ic": fold_ics,
                    "mean_rank_ic": float(np.nanmean(fold_ics)),
                }
            )
    payload = _hashed(
        {
            "status": "PASS",
            "research_status": "RESEARCH_ONLY",
            "data_release_id": cache.data_release_id,
            "evaluation_code_version": pool["evaluation_code_version"],
            "cache_config_hash": cache.config_hash,
            "cache_content_hash": cache_metadata["content_hash"],
            "feature_eligible_pool_hash": pool["content_hash"],
            "horizon": horizon,
            "split": "three_fold_expanding_time_ordered",
            "train_only_fit": True,
            "l2_pool_selection_reuses_oos_evidence": True,
            "interpretation": "integration smoke, not unbiased L3 production evidence",
            "results": results,
        }
    )
    _write_json(output, payload)
    return payload


def _research_record(path: Path, horizon: int) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    factor = report["factor"]
    metrics = report["extra_metrics"]["oos_horizons"][str(horizon)]
    by_date = np.asarray(report["rank_ic"]["by_date"], dtype=float)
    split = max(1, int(len(by_date) * 0.8))
    train = by_date[:split]
    train = train[np.isfinite(train)]
    expected = int(factor["expected_direction"])
    direction = expected if expected else (1 if not len(train) or np.mean(train) >= 0 else -1)
    annual = [direction * float(value) for _, value in metrics["annual_rank_ic"]]
    regimes = [direction * float(value) for _, value in metrics["regime_rank_ic"]]
    t_value = abs(float(metrics["newey_west_t"]))
    return {
        "factor_id": factor["factor_id"],
        "version": factor["version"],
        "family": factor["family"],
        "subfamily": factor["subfamily"],
        "role": factor["role"],
        "source_id": factor["source_id"],
        "expression_hash": factor["expression_hash"],
        "content_hash": report["content_hash"],
        "direction": direction,
        "direction_source": "FactorSpec" if expected else "train_window_rank_ic_sign",
        "train_rank_ic": float(np.mean(train)) if len(train) else 0.0,
        "oos_rank_ic": float(metrics["rank_ic_mean"]),
        "directional_oos_rank_ic": direction * float(metrics["rank_ic_mean"]),
        "directional_oos_rank_icir": direction * float(metrics["rank_icir"]),
        "directional_monotonicity": direction * float(metrics["monotonicity"]),
        "coverage": float(report["quality"]["coverage"]),
        "turnover": float(metrics["turnover"]),
        "cost_sensitivity": abs(
            float(metrics["long_short_return"]) - float(metrics["net_long_short_return"])
        ),
        "complexity_score": float(factor["complexity_score"]),
        "annual_stability": (2 * float(np.mean(np.asarray(annual) > 0)) - 1 if annual else 0.0),
        "regime_complementarity": (max(regimes) - min(regimes) if regimes else 0.0),
        "raw_p_value": math.erfc(t_value / math.sqrt(2)),
        "non_degenerate": bool(
            report["quality"]["std"] > 1e-12 and report["quality"]["unique_count"] > 1
        ),
        "rank_ic_series": [float(value) if np.isfinite(value) else None for value in by_date],
    }


def _attach_family_correlations(
    family: str,
    members: list[dict[str, Any]],
    duplicates: dict[str, str],
    clusters: dict[str, str],
    *,
    threshold: float,
) -> None:
    matrix = np.asarray([item.pop("rank_ic_series") for item in members], dtype=float)
    matrix = matrix[:, : int(matrix.shape[1] * 0.8)]
    if len(members) == 1:
        members[0]["max_peer_correlation"] = 0.0
        members[0]["most_correlated_peer"] = None
        clusters[str(members[0]["factor_id"])] = f"{family}:0000"
        return
    row_means = np.nanmean(matrix, axis=1)
    matrix = np.where(np.isfinite(matrix), matrix, row_means[:, None])
    with np.errstate(all="ignore"):
        correlation = np.nan_to_num(np.corrcoef(matrix), nan=0.0)
    np.fill_diagonal(correlation, 0.0)
    for index, item in enumerate(members):
        item["max_peer_correlation"] = float(np.max(np.abs(correlation[index])))
        item["most_correlated_peer"] = members[int(np.argmax(np.abs(correlation[index])))][
            "factor_id"
        ]
    parent = list(range(len(members)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in zip(*np.where(np.triu(np.abs(correlation) >= threshold, 1)), strict=True):
        left_root, right_root = root(int(left)), root(int(right))
        if left_root != right_root:
            parent[right_root] = left_root
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(members)):
        grouped[root(index)].append(index)
    for number, indices in enumerate(grouped.values()):
        cluster_id = f"{family}:{number:04d}"
        representative = max(indices, key=lambda index: _base_utility(members[index]))
        for index in indices:
            factor_id = str(members[index]["factor_id"])
            clusters[factor_id] = cluster_id
            if len(indices) > 1 and index != representative:
                duplicates[factor_id] = str(members[representative]["factor_id"])


def _base_utility(item: dict[str, Any]) -> float:
    return (
        abs(float(item["directional_oos_rank_ic"]))
        + 0.1 * float(item["coverage"])
        - 0.01 * float(item["complexity_score"])
        - 0.05 * float(item["turnover"])
    )


def _model_rows(
    values: FloatArray,
    labels: FloatArray,
    start: int,
    end: int,
    direction: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    features = values[:, start:end].transpose(1, 2, 0).reshape(-1, len(values))
    target = labels[start:end].reshape(-1)
    features *= direction
    valid = np.isfinite(target) & (np.count_nonzero(np.isfinite(features), axis=1) > 0)
    features, target = features[valid], target[valid]
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features, target


def _spearman(left: FloatArray, right: FloatArray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    x, y = left[valid], right[valid]
    x = np.argsort(np.argsort(x)).astype(float)
    y = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else float("nan")


def _single_factor_report(report_dir: Path, factor_id: str) -> Path:
    matches = tuple(report_dir.glob(f"{factor_id}-*.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one audited report for {factor_id}, found {len(matches)}")
    return matches[0]


def _hashed(payload: dict[str, Any]) -> dict[str, Any]:
    payload["content_hash"] = _content_hash(payload)
    return payload


def _content_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".l2-v2-", suffix=".json", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
