#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from aquant.strategies.microcap.experiments import (  # type: ignore[import-untyped]
    RebalanceFrequency,
)
from scripts.audit_nested_l3_l4_root_causes import _simulate_control


def main() -> None:
    args = _arguments()
    cache = json.loads((args.cache_dir / "metadata.json").read_text())
    l3 = json.loads(args.l3_metadata.read_text())
    l4 = json.loads(args.l4_report.read_text())
    attribution = json.loads(args.l4_attribution.read_text())
    eligibility = json.loads((args.eligibility_dir / "eligibility_metadata.json").read_text())
    dates = tuple(date.fromisoformat(value) for value in cache["trade_dates"])
    scores = np.load(args.l3_scores, mmap_mode="r")
    adjusted_open = np.load(args.market_matrices / "adjusted_open.npy", mmap_mode="r")
    mask = np.load(args.eligibility_dir / "production_eligibility_mask.npy", mmap_mode="r")
    if scores.shape != adjusted_open.shape or scores.shape != mask.shape:
        raise ValueError("execution bridge inputs do not align")
    if eligibility["cache_content_hash"] != cache["content_hash"]:
        raise ValueError("eligibility and convergence cache conflict")
    benchmark = _benchmark_returns(args.benchmark_proxy)
    fold_actual = {int(row["fold"]): row for row in _csv_rows(args.l4_fold_attribution)}
    rows: list[dict[str, Any]] = []
    aggregate_paths: dict[str, list[float]] = {
        "all_signal_gross": [],
        "eligible_signal_gross": [],
        "eligible_signal_10bps": [],
        "benchmark": [],
    }
    for fold in l3["folds"]:
        selected = fold["l4_selection"]["selected"]
        paths = _paths(
            scores=scores,
            adjusted_open=adjusted_open,
            mask=mask,
            dates=dates,
            fold=fold,
            benchmark=benchmark,
            target_count=int(selected["target_count"]),
            frequency=RebalanceFrequency(selected["frequency"]),
        )
        actual = fold_actual[int(fold["fold"])]
        row = _bridge_row(
            fold=int(fold["fold"]),
            start=fold["test_start"],
            end=fold["test_end"],
            target_count=int(selected["target_count"]),
            frequency=str(selected["frequency"]),
            paths=paths,
            actual_gross_excess=float(actual["conditional_gross_annual_excess_return"]),
            actual_net_excess=float(actual["net_annual_excess_return"]),
        )
        rows.append(row)
        for key in aggregate_paths:
            aggregate_paths[key].extend(paths[key])
    rows.append(
        _bridge_row(
            fold="ALL",
            start=l4["start_date"],
            end=l4["end_date"],
            target_count="FOLD_SELECTED",
            frequency="FOLD_SELECTED",
            paths=aggregate_paths,
            actual_gross_excess=float(attribution["gross_annual_excess_return"]),
            actual_net_excess=float(l4["performance"]["annual_excess_return"]),
        )
    )
    stability_rows = _configuration_stability(
        scores=scores,
        adjusted_open=adjusted_open,
        mask=mask,
        dates=dates,
        l3=l3,
        benchmark=benchmark,
    )
    text = _csv_text(rows)
    stability_text = _csv_text(stability_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "selected_configuration_execution_bridge.csv", text)
    _write(args.output_dir / "configuration_selection_stability.csv", stability_text)
    aggregate = rows[-1]
    stable = {
        "status": "PASS_RESEARCH_ONLY",
        "stage": "SELECTED_CONFIGURATION_EXECUTION_BRIDGE",
        "diagnostic_only": True,
        "outer_oos_used_for_retuning": False,
        "aggregate": aggregate,
        "configuration_selection_stability": _stability_summary(stability_rows),
        "definitions": {
            "universe_filter_effect": "eligible ideal gross - all-score ideal gross",
            "execution_constraint_effect": "actual conditional gross - eligible ideal gross",
            "actual_cost_effect": "actual net - actual conditional gross",
        },
        "limitations": [
            "Effects are sequential path differences, not permutation-invariant Shapley values.",
            "The production eligibility mask is applied only on rebalance signal dates.",
            "Outer OOS has already been observed and is diagnostic only.",
        ],
        "csv_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "stability_csv_sha256": hashlib.sha256(stability_text.encode()).hexdigest(),
        "provenance": {
            "cache_content_hash": cache["content_hash"],
            "l3_content_hash": l3["content_hash"],
            "l4_content_hash": l4["content_hash"],
            "attribution_content_hash": attribution["content_hash"],
            "eligibility_content_hash": eligibility["content_hash"],
            "code_version": args.code_version,
        },
    }
    stable["content_hash"] = _hash(stable)
    payload = {**stable, "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
    _write(args.output_dir / "execution_bridge_summary.json", json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge ideal L3 signal to realized L4 returns")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--market-matrices", type=Path, required=True)
    parser.add_argument("--eligibility-dir", type=Path, required=True)
    parser.add_argument("--l3-scores", type=Path, required=True)
    parser.add_argument("--l3-metadata", type=Path, required=True)
    parser.add_argument("--l4-report", type=Path, required=True)
    parser.add_argument("--l4-attribution", type=Path, required=True)
    parser.add_argument("--l4-fold-attribution", type=Path, required=True)
    parser.add_argument("--benchmark-proxy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    return parser.parse_args()


def _paths(
    *,
    scores: np.ndarray[Any, Any],
    adjusted_open: np.ndarray[Any, Any],
    mask: np.ndarray[Any, Any],
    dates: tuple[date, ...],
    fold: dict[str, Any],
    benchmark: dict[date, float],
    target_count: int,
    frequency: RebalanceFrequency,
) -> dict[str, list[float]]:
    common = {
        "scores": scores,
        "prices": adjusted_open,
        "dates": dates,
        "folds": [fold],
        "benchmark": benchmark,
        "target_count": target_count,
        "frequency": frequency,
    }
    all_gross = _simulate_control(**common, cost_bps=0.0)
    eligible_gross = _simulate_control(**common, cost_bps=0.0, eligibility_mask=mask)
    eligible_cost = _simulate_control(**common, cost_bps=10.0, eligibility_mask=mask)
    return {
        "all_signal_gross": [float(item["strategy_return"]) for item in all_gross],
        "eligible_signal_gross": [float(item["strategy_return"]) for item in eligible_gross],
        "eligible_signal_10bps": [float(item["strategy_return"]) for item in eligible_cost],
        "benchmark": [float(item["benchmark_return"]) for item in all_gross],
    }


def _bridge_row(
    *,
    fold: int | str,
    start: str,
    end: str,
    target_count: int | str,
    frequency: str,
    paths: dict[str, list[float]],
    actual_gross_excess: float,
    actual_net_excess: float,
) -> dict[str, Any]:
    benchmark = _annualized(paths["benchmark"])
    all_gross = _annualized(paths["all_signal_gross"]) - benchmark
    eligible_gross = _annualized(paths["eligible_signal_gross"]) - benchmark
    eligible_cost = _annualized(paths["eligible_signal_10bps"]) - benchmark
    return {
        "fold": fold,
        "test_start": start,
        "test_end": end,
        "sessions": len(paths["benchmark"]),
        "target_count": target_count,
        "frequency": frequency,
        "all_signal_gross_excess": all_gross,
        "eligible_signal_gross_excess": eligible_gross,
        "eligible_signal_10bps_excess": eligible_cost,
        "actual_conditional_gross_excess": actual_gross_excess,
        "actual_net_excess": actual_net_excess,
        "universe_filter_effect": eligible_gross - all_gross,
        "proxy_10bps_effect": eligible_cost - eligible_gross,
        "execution_constraint_effect": actual_gross_excess - eligible_gross,
        "actual_cost_effect": actual_net_excess - actual_gross_excess,
        "total_conversion_effect": actual_net_excess - all_gross,
    }


def _configuration_stability(
    *,
    scores: np.ndarray[Any, Any],
    adjusted_open: np.ndarray[Any, Any],
    mask: np.ndarray[Any, Any],
    dates: tuple[date, ...],
    l3: dict[str, Any],
    benchmark: dict[date, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in l3["folds"]:
        candidates: list[dict[str, Any]] = fold["l4_selection"]["candidates"]
        selected = fold["l4_selection"]["selected"]
        fold_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            daily = _simulate_control(
                scores=scores,
                prices=adjusted_open,
                dates=dates,
                folds=[fold],
                benchmark=benchmark,
                target_count=int(candidate["target_count"]),
                frequency=RebalanceFrequency(candidate["frequency"]),
                cost_bps=10.0,
                eligibility_mask=mask,
            )
            strategy = [float(item["strategy_return"]) for item in daily]
            benchmark_path = [float(item["benchmark_return"]) for item in daily]
            fold_rows.append(
                {
                    "fold": int(fold["fold"]),
                    "target_count": int(candidate["target_count"]),
                    "frequency": str(candidate["frequency"]),
                    "selected": bool(
                        candidate["target_count"] == selected["target_count"]
                        and candidate["frequency"] == selected["frequency"]
                    ),
                    "inner_selection_score": float(candidate["selection_score"]),
                    "inner_annual_excess": float(candidate["annual_excess_return"]),
                    "outer_eligible_10bps_excess": _annualized(strategy)
                    - _annualized(benchmark_path),
                }
            )
        inner_order = sorted(
            range(len(fold_rows)),
            key=lambda index: float(fold_rows[index]["inner_selection_score"]),
            reverse=True,
        )
        outer_order = sorted(
            range(len(fold_rows)),
            key=lambda index: float(fold_rows[index]["outer_eligible_10bps_excess"]),
            reverse=True,
        )
        inner_rank = {index: rank for rank, index in enumerate(inner_order, start=1)}
        outer_rank = {index: rank for rank, index in enumerate(outer_order, start=1)}
        for index, row in enumerate(fold_rows):
            row["inner_rank"] = inner_rank[index]
            row["outer_rank"] = outer_rank[index]
            rows.append(row)
    return rows


def _stability_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if bool(row["selected"])]
    folds = sorted({int(row["fold"]) for row in rows})
    correlations = []
    selected_minus_candidate_mean = []
    for fold in folds:
        members = [row for row in rows if int(row["fold"]) == fold]
        inner = np.asarray([row["inner_selection_score"] for row in members], dtype=float)
        outer = np.asarray([row["outer_eligible_10bps_excess"] for row in members], dtype=float)
        correlations.append(float(np.corrcoef(inner, outer)[0, 1]))
        chosen = next(row for row in members if bool(row["selected"]))
        selected_minus_candidate_mean.append(
            float(chosen["outer_eligible_10bps_excess"]) - float(np.mean(outer))
        )
    chosen_inner = np.asarray([row["inner_annual_excess"] for row in selected], dtype=float)
    chosen_outer = np.asarray([row["outer_eligible_10bps_excess"] for row in selected], dtype=float)
    return {
        "fold_count": len(folds),
        "trials_per_fold": len(rows) // len(folds),
        "selected_outer_ranks": [int(row["outer_rank"]) for row in selected],
        "mean_selected_outer_rank": float(np.mean([row["outer_rank"] for row in selected])),
        "mean_within_fold_inner_outer_correlation": float(np.mean(correlations)),
        "within_fold_inner_outer_correlations": correlations,
        "mean_selected_minus_candidate_outer_excess": float(np.mean(selected_minus_candidate_mean)),
        "chosen_inner_outer_correlation": (
            float(np.corrcoef(chosen_inner, chosen_outer)[0, 1])
            if len(chosen_inner) > 1
            else math.nan
        ),
        "chosen_mean_optimism": float(np.mean(chosen_inner - chosen_outer)),
        "chosen_mean_absolute_error": float(np.mean(np.abs(chosen_inner - chosen_outer))),
    }


def _annualized(returns: list[float]) -> float:
    total = math.prod(1 + value for value in returns)
    return total ** (252 / len(returns)) - 1 if total > 0 else -1.0


def _benchmark_returns(path: Path) -> dict[date, float]:
    return {
        date.fromisoformat(row["trade_date"]): float(row["return_rate"]) for row in _csv_rows(path)
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _csv_text(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


if __name__ == "__main__":
    main()
