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

from aquant.factors.operators.cross_sectional import cs_rank  # type: ignore[import-untyped]
from scripts.audit_nested_l3_l4_root_causes import _block_bootstrap_mean


def main() -> None:
    args = _arguments()
    cache = json.loads((args.cache_dir / "metadata.json").read_text())
    original_metadata = json.loads(args.original_metadata.read_text())
    adjusted_metadata = json.loads(args.adjusted_metadata.read_text())
    dates = tuple(date.fromisoformat(value) for value in cache["trade_dates"])
    original_scores = np.load(args.original_scores, mmap_mode="r")
    adjusted_scores = np.load(args.adjusted_scores, mmap_mode="r")
    if original_scores.shape != adjusted_scores.shape:
        raise ValueError("L3 label variant score matrices do not align")
    original_daily = _csv_by_date(args.original_audit / "daily_signal_diagnostics.csv")
    adjusted_daily = _csv_by_date(args.adjusted_audit / "daily_signal_diagnostics.csv")
    positions = _outer_positions(dates, original_metadata)
    score_rows = _score_correlations(
        dates, positions, original_scores, adjusted_scores, original_daily
    )
    metric_rows = _metric_comparison(original_daily, adjusted_daily)
    control_rows = _control_comparison(
        args.original_audit / "control_matrix.csv",
        args.adjusted_audit / "control_matrix.csv",
    )
    model_rows = _model_comparison(original_metadata, adjusted_metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "label_variant_score_correlations.csv": _csv_text(score_rows),
        "label_variant_metric_deltas.csv": _csv_text(metric_rows),
        "label_variant_control_deltas.csv": _csv_text(control_rows),
        "label_variant_model_changes.csv": _csv_text(model_rows),
    }
    evidence = {}
    for name, text in outputs.items():
        _write(args.output_dir / name, text)
        evidence[name] = hashlib.sha256(text.encode()).hexdigest()
    stable = {
        "status": "PASS_RESEARCH_ONLY",
        "stage": "NESTED_L3_LABEL_VARIANT_COMPARISON",
        "diagnostic_only": True,
        "outer_oos_used_for_retuning": False,
        "summary": _summary(score_rows, metric_rows, control_rows, model_rows),
        "artifacts": evidence,
        "provenance": {
            "cache_content_hash": cache["content_hash"],
            "original_l3_content_hash": original_metadata["content_hash"],
            "adjusted_l3_content_hash": adjusted_metadata["content_hash"],
            "code_version": args.code_version,
        },
    }
    stable["content_hash"] = _hash(stable)
    payload = {**stable, "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
    _write(args.output_dir / "label_variant_comparison.json", json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare raw- and adjusted-label nested L3")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--original-scores", type=Path, required=True)
    parser.add_argument("--adjusted-scores", type=Path, required=True)
    parser.add_argument("--original-metadata", type=Path, required=True)
    parser.add_argument("--adjusted-metadata", type=Path, required=True)
    parser.add_argument("--original-audit", type=Path, required=True)
    parser.add_argument("--adjusted-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    return parser.parse_args()


def _outer_positions(dates: tuple[date, ...], metadata: dict[str, Any]) -> list[tuple[int, int]]:
    index = {value: position for position, value in enumerate(dates)}
    result: list[tuple[int, int]] = []
    for fold in metadata["folds"]:
        start = index[date.fromisoformat(fold["test_start"])]
        end = index[date.fromisoformat(fold["test_end"])] + 1
        result.extend((int(fold["fold"]), position) for position in range(start, end))
    return result


def _score_correlations(
    dates: tuple[date, ...],
    positions: list[tuple[int, int]],
    original: np.ndarray[Any, Any],
    adjusted: np.ndarray[Any, Any],
    daily: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for fold, position in positions:
        left = np.asarray(original[position], dtype=float)
        right = np.asarray(adjusted[position], dtype=float)
        valid = np.isfinite(left) & np.isfinite(right)
        correlation = math.nan
        if np.count_nonzero(valid) > 1:
            correlation = float(np.corrcoef(cs_rank(left[valid]), cs_rank(right[valid]))[0, 1])
        trade_date = dates[position].isoformat()
        rows.append(
            {
                "trade_date": trade_date,
                "fold": fold,
                "score_rank_correlation": correlation,
                "common_observations": int(np.count_nonzero(valid)),
                "original_topk_excess": float(daily[trade_date]["adjusted_open_topk_excess"]),
            }
        )
    return rows


def _metric_comparison(
    original: dict[str, dict[str, str]], adjusted: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    fields = (
        "adjusted_open_rank_ic",
        "adjusted_open_topk_excess",
        "adjusted_open_tail_rank_ic",
        "adjusted_open_long_short_decile",
    )
    rows = []
    common = sorted(set(original) & set(adjusted))
    for offset, field in enumerate(fields):
        differences = np.asarray(
            [float(adjusted[day][field]) - float(original[day][field]) for day in common],
            dtype=float,
        )
        finite = differences[np.isfinite(differences)]
        low, high, p_value = _block_bootstrap_mean(
            finite, block_size=21, replicates=2000, seed=20260850 + offset
        )
        rows.append(
            {
                "metric": field,
                "observations": len(finite),
                "adjusted_minus_original_mean": float(np.mean(finite)),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_two_sided_p": p_value,
            }
        )
    return rows


def _control_comparison(original_path: Path, adjusted_path: Path) -> list[dict[str, Any]]:
    def key(row: dict[str, str]) -> tuple[str, str, str, str]:
        return (
            row["target_count"],
            row["frequency"],
            row["price_basis"],
            row["cost_bps"],
        )

    original = {key(row): row for row in _csv_rows(original_path)}
    adjusted = {key(row): row for row in _csv_rows(adjusted_path)}
    rows = []
    for item in sorted(set(original) & set(adjusted)):
        left = original[item]
        right = adjusted[item]
        rows.append(
            {
                "target_count": item[0],
                "frequency": item[1],
                "price_basis": item[2],
                "cost_bps": item[3],
                "original_annual_excess": float(left["annual_excess_return"]),
                "adjusted_annual_excess": float(right["annual_excess_return"]),
                "adjusted_minus_original": float(right["annual_excess_return"])
                - float(left["annual_excess_return"]),
            }
        )
    return rows


def _model_comparison(original: dict[str, Any], adjusted: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for left_fold, right_fold in zip(original["folds"], adjusted["folds"], strict=True):
        right_families = {item["family"]: item for item in right_fold["families"]}
        for left in left_fold["families"]:
            right = right_families[left["family"]]
            rows.append(
                {
                    "fold": int(left_fold["fold"]),
                    "family": left["family"],
                    "original_method": left["selected_method"],
                    "adjusted_method": right["selected_method"],
                    "method_changed": left["selected_method"] != right["selected_method"],
                    "original_outer_rank_ic": left["outer_rank_ic"],
                    "adjusted_outer_rank_ic": right["outer_rank_ic"],
                }
            )
    return rows


def _summary(
    scores: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    models: list[dict[str, Any]],
) -> dict[str, Any]:
    metric = {row["metric"]: row for row in metrics}
    control_deltas = np.asarray([row["adjusted_minus_original"] for row in controls], dtype=float)
    return {
        "mean_daily_score_rank_correlation": float(
            np.nanmean([row["score_rank_correlation"] for row in scores])
        ),
        "minimum_daily_score_rank_correlation": float(
            np.nanmin([row["score_rank_correlation"] for row in scores])
        ),
        "method_changes": sum(bool(row["method_changed"]) for row in models),
        "family_models": len(models),
        "rank_ic_effect": metric["adjusted_open_rank_ic"]["adjusted_minus_original_mean"],
        "topk_5d_excess_effect": metric["adjusted_open_topk_excess"][
            "adjusted_minus_original_mean"
        ],
        "mean_fixed_control_annual_excess_effect": float(np.mean(control_deltas)),
        "minimum_fixed_control_annual_excess_effect": float(np.min(control_deltas)),
        "maximum_fixed_control_annual_excess_effect": float(np.max(control_deltas)),
    }


def _csv_by_date(path: Path) -> dict[str, dict[str, str]]:
    return {row["trade_date"]: row for row in _csv_rows(path)}


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
