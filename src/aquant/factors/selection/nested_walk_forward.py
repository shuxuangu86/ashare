from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from aquant.factors.evaluation.multiple_testing import benjamini_hochberg

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OuterFold:
    fold: int
    test_start: date
    test_end: date
    test_observations: int


def define_outer_folds(
    trade_dates: tuple[date, ...], *, fold_count: int = 5
) -> tuple[OuterFold, ...]:
    """Split the locked trading period into contiguous, non-overlapping outer folds."""
    if fold_count < 2 or len(trade_dates) < fold_count:
        raise ValueError("outer walk-forward requires at least two non-empty folds")
    if tuple(sorted(set(trade_dates))) != trade_dates:
        raise ValueError("outer trade dates must be unique and ordered")
    positions = np.array_split(np.arange(len(trade_dates)), fold_count)
    return tuple(
        OuterFold(
            fold=index,
            test_start=trade_dates[int(part[0])],
            test_end=trade_dates[int(part[-1])],
            test_observations=len(part),
        )
        for index, part in enumerate(positions)
    )


def build_nested_l2_pools(
    *,
    factor_records: dict[str, dict[str, Any]],
    evidence_dates: tuple[date, ...],
    outer_dates: tuple[date, ...],
    output: Path,
    purge_observations: int = 6,
    fold_count: int = 5,
    maximum_family_members: int = 9,
    correlation_threshold: float = 0.995,
    fdr_alpha: float = 0.05,
    minimum_observations: int = 126,
    minimum_daily_ic_coverage: float = 0.60,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select each outer fold's L2 pool using only purged preceding RankIC evidence."""
    if purge_observations < 6:
        raise ValueError("five-day T+1 labels require at least six purged observations")
    if maximum_family_members < 5:
        raise ValueError("maximum_family_members must allow five family archetypes")
    if tuple(sorted(set(evidence_dates))) != evidence_dates:
        raise ValueError("evidence dates must be unique and ordered")
    if not factor_records:
        raise ValueError("factor records are empty")
    if any(len(item["rank_ic"]) != len(evidence_dates) for item in factor_records.values()):
        raise ValueError("every RankIC series must align with evidence_dates")
    folds = define_outer_folds(outer_dates, fold_count=fold_count)
    all_dates = np.asarray(evidence_dates, dtype=object)
    fold_payloads: list[dict[str, Any]] = []
    union: set[str] = set()
    for fold in folds:
        visible = np.flatnonzero(all_dates < fold.test_start)
        if len(visible) <= purge_observations:
            raise ValueError(f"fold {fold.fold} has insufficient history before purge")
        train_positions = visible[:-purge_observations]
        train_dates = tuple(evidence_dates[int(position)] for position in train_positions)
        selected, diagnostics = _select_fold(
            factor_records,
            train_dates=train_dates,
            train_positions=train_positions,
            maximum_family_members=maximum_family_members,
            correlation_threshold=correlation_threshold,
            fdr_alpha=fdr_alpha,
            minimum_observations=minimum_observations,
            minimum_daily_ic_coverage=minimum_daily_ic_coverage,
        )
        union.update(selected)
        fold_payloads.append(
            {
                "fold": fold.fold,
                "train_start": train_dates[0].isoformat(),
                "train_end_after_purge": train_dates[-1].isoformat(),
                "train_observations": len(train_dates),
                "purged_observations": purge_observations,
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "test_observations": fold.test_observations,
                "outer_test_used_for_selection": False,
                "selected_count": len(selected),
                "selected_factor_ids": sorted(selected),
                "factors": selected,
                "diagnostics": diagnostics,
            }
        )
    payload: dict[str, Any] = {
        "status": "PASS",
        "research_status": "RESEARCH_ONLY",
        "stage": "NESTED_WALK_FORWARD_L2_SELECTION",
        "neutralization": "SIZE_NEUTRAL",
        "fold_count": fold_count,
        "purge_observations": purge_observations,
        "fdr_alpha": fdr_alpha,
        "correlation_threshold": correlation_threshold,
        "direction_policy": "SOURCE_EXPECTED_DIRECTION_ELSE_TRAIN_ONLY_RANK_IC_SIGN",
        "future_aggregate_metrics_used": False,
        "outer_test_used_for_selection": False,
        "folds": fold_payloads,
        "union_factor_count": len(union),
        "union_factor_ids": sorted(union),
        "provenance": provenance or {},
    }
    payload["content_hash"] = _content_hash(payload)
    _write_json(output, payload)
    return payload


def _select_fold(
    records: dict[str, dict[str, Any]],
    *,
    train_dates: tuple[date, ...],
    train_positions: npt.NDArray[np.int64],
    maximum_family_members: int,
    correlation_threshold: float,
    fdr_alpha: float,
    minimum_observations: int,
    minimum_daily_ic_coverage: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    rejected: dict[str, str] = {}
    for factor_id, record in records.items():
        values = np.asarray(record["rank_ic"], dtype=np.float64)[train_positions]
        finite = np.isfinite(values)
        observation_count = int(finite.sum())
        coverage = observation_count / len(values)
        if observation_count < minimum_observations:
            rejected[factor_id] = "INSUFFICIENT_TRAIN_RANK_IC_OBSERVATIONS"
            continue
        if coverage < minimum_daily_ic_coverage:
            rejected[factor_id] = "INSUFFICIENT_TRAIN_DAILY_IC_COVERAGE"
            continue
        clean = values[finite]
        raw_mean = float(np.mean(clean))
        expected = int(record.get("expected_direction", 0))
        direction = expected if expected in {-1, 1} else (1 if raw_mean >= 0 else -1)
        directional = clean * direction
        mean = float(np.mean(directional))
        standard_deviation = float(np.std(directional, ddof=1))
        if not np.isfinite(standard_deviation) or standard_deviation <= 1e-12:
            rejected[factor_id] = "DEGENERATE_TRAIN_RANK_IC"
            continue
        t_statistic, p_value, hac_lags = _hac_test(directional, maximum_lags=5)
        annual_means = _annual_means(train_dates, values, direction)
        candidates[factor_id] = {
            "factor_id": factor_id,
            "family": str(record["family"]),
            "subfamily": str(record.get("subfamily", "unknown")),
            "expression_hash": str(record["expression_hash"]),
            "complexity_score": float(record.get("complexity_score", 100.0)),
            "direction": direction,
            "direction_source": "SOURCE" if expected in {-1, 1} else "TRAIN_ONLY_RANK_IC_SIGN",
            "train_rank_ic": mean,
            "train_rank_icir": mean / standard_deviation,
            "train_t_statistic": t_statistic,
            "raw_p_value": p_value,
            "p_value_method": "NEWEY_WEST_HAC_NORMAL_APPROXIMATION",
            "serial_correlation_lags": hac_lags,
            "train_daily_ic_coverage": coverage,
            "train_observations": observation_count,
            "annual_stability": sum(float(value[1]) >= 0 for value in annual_means)
            / len(annual_means),
            "annual_directional_rank_ic": annual_means,
            "rank_ic_series": values * direction,
        }
    if not candidates:
        raise ValueError("training window produced no valid L2 candidates")
    ordered_ids = sorted(candidates)
    adjusted = benjamini_hochberg(
        tuple(float(candidates[item]["raw_p_value"]) for item in ordered_ids), alpha=fdr_alpha
    )
    for factor_id, (adjusted_p, significant) in zip(ordered_ids, adjusted, strict=True):
        candidates[factor_id]["fdr_adjusted_p_value"] = adjusted_p
        candidates[factor_id]["fdr_significant"] = significant
    duplicates = _deduplicate(candidates, threshold=correlation_threshold)
    family_members: dict[str, list[str]] = defaultdict(list)
    for factor_id, item in candidates.items():
        if factor_id not in duplicates:
            family_members[str(item["family"])].append(factor_id)
    selected: dict[str, dict[str, Any]] = {}
    for members in family_members.values():
        peer_correlations = _peer_correlations(candidates, members)
        for factor_id, value in peer_correlations.items():
            candidates[factor_id]["max_peer_rank_ic_correlation"] = value
        reasons: dict[str, set[str]] = defaultdict(set)
        reasons[max(members, key=lambda item: candidates[item]["train_rank_ic"])].add(
            "TRAIN_STRONGEST"
        )
        reasons[max(members, key=lambda item: candidates[item]["annual_stability"])].add(
            "TRAIN_STABLE"
        )
        reasons[min(members, key=lambda item: candidates[item]["complexity_score"])].add(
            "LOW_COMPLEXITY"
        )
        reasons[
            min(members, key=lambda item: candidates[item]["max_peer_rank_ic_correlation"])
        ].add("TRAIN_MOST_INDEPENDENT")
        reasons[min(members, key=lambda item: candidates[item]["fdr_adjusted_p_value"])].add(
            "FAMILY_BEST_FDR"
        )
        for factor_id in sorted(members, key=lambda item: _utility(candidates[item]), reverse=True):
            if len(reasons) >= maximum_family_members:
                break
            reasons[factor_id].add("TRAIN_PARETO_UTILITY")
        for factor_id, factor_reasons in reasons.items():
            item = {
                key: value
                for key, value in candidates[factor_id].items()
                if key != "rank_ic_series"
            }
            item["reason_codes"] = sorted(factor_reasons)
            selected[factor_id] = item
    diagnostics = {
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "rejected_count": len(rejected),
        "duplicate_count": len(duplicates),
        "fdr_significant_count": sum(bool(item["fdr_significant"]) for item in candidates.values()),
        "family_count": len(family_members),
        "duplicate_of": dict(sorted(duplicates.items())),
        "rejection_reason_counts": dict(
            sorted(
                (reason, list(rejected.values()).count(reason)) for reason in set(rejected.values())
            )
        ),
    }
    return selected, diagnostics


def _hac_test(values: npt.ArrayLike, *, maximum_lags: int) -> tuple[float, float, int]:
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 2 or maximum_lags < 0:
        raise ValueError("HAC test requires finite observations and non-negative lags")
    centered = sample - np.mean(sample)
    lags = min(maximum_lags, len(sample) - 1)
    long_run_variance = float(np.dot(centered, centered) / len(sample))
    for lag in range(1, lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / len(sample))
        long_run_variance += 2 * (1 - lag / (lags + 1)) * covariance
    variance_of_mean = max(long_run_variance, 0.0) / len(sample)
    if variance_of_mean <= 1e-24:
        statistic = (
            0.0
            if np.isclose(np.mean(sample), 0.0)
            else math.copysign(float("inf"), float(np.mean(sample)))
        )
    else:
        statistic = float(np.mean(sample) / math.sqrt(variance_of_mean))
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return statistic, p_value, lags


def _annual_means(
    dates: tuple[date, ...], values: FloatArray, direction: int
) -> list[list[float | int]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for current_date, value in zip(dates, values, strict=True):
        if np.isfinite(value):
            grouped[current_date.year].append(float(value) * direction)
    return [[year, float(np.mean(grouped[year]))] for year in sorted(grouped)]


def _deduplicate(records: dict[str, dict[str, Any]], *, threshold: float) -> dict[str, str]:
    duplicates: dict[str, str] = {}
    expression_groups: dict[str, list[str]] = defaultdict(list)
    for factor_id, item in records.items():
        expression_groups[str(item["expression_hash"])].append(factor_id)
    for members in expression_groups.values():
        representative = max(members, key=lambda item: _utility(records[item]))
        duplicates.update({item: representative for item in members if item != representative})
    family_groups: dict[str, list[str]] = defaultdict(list)
    for factor_id, item in records.items():
        if factor_id not in duplicates:
            family_groups[str(item["family"])].append(factor_id)
    for members in family_groups.values():
        representatives: list[str] = []
        for factor_id in sorted(members, key=lambda item: _utility(records[item]), reverse=True):
            duplicate = next(
                (
                    kept
                    for kept in representatives
                    if abs(
                        _pair_correlation(
                            records[factor_id]["rank_ic_series"],
                            records[kept]["rank_ic_series"],
                        )
                    )
                    >= threshold
                ),
                None,
            )
            if duplicate is None:
                representatives.append(factor_id)
            else:
                duplicates[factor_id] = duplicate
    return duplicates


def _peer_correlations(records: dict[str, dict[str, Any]], members: list[str]) -> dict[str, float]:
    result = {factor_id: 0.0 for factor_id in members}
    for index, left in enumerate(members):
        for right in members[index + 1 :]:
            correlation = abs(
                _pair_correlation(records[left]["rank_ic_series"], records[right]["rank_ic_series"])
            )
            result[left] = max(result[left], correlation)
            result[right] = max(result[right], correlation)
    return result


def _pair_correlation(left: FloatArray, right: FloatArray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 20:
        return 0.0
    left_valid = left[valid]
    right_valid = right[valid]
    if np.std(left_valid) <= 1e-12 or np.std(right_valid) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_valid, right_valid)[0, 1])


def _utility(item: dict[str, Any]) -> float:
    return (
        float(item["train_rank_ic"])
        + 0.4 * float(item["train_rank_icir"])
        + 0.15 * float(item["annual_stability"])
        + 0.05 * float(item["train_daily_ic_coverage"])
        - 0.002 * float(item["complexity_score"])
        - 0.05 * float(item.get("max_peer_rank_ic_correlation", 0.0))
    )


def _content_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".nested-l2-", suffix=".json", dir=path.parent)
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
