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
    generate_rebalance_dates,
)
from scripts.audit_nested_l3_l4_root_causes import _simulate_control

_FEATURES = (
    ("PIT_UNIVERSE", 1),
    ("LISTED_120_DAYS", 2),
    ("NOT_ST", 4),
    ("NOT_SUSPENDED", 8),
    ("NOT_DELISTING_RISK", 16),
)


def main() -> None:
    args = _arguments()
    cache = json.loads((args.cache_dir / "metadata.json").read_text())
    l3 = json.loads(args.l3_metadata.read_text())
    eligibility = json.loads((args.eligibility_dir / "eligibility_metadata.json").read_text())
    if eligibility["cache_content_hash"] != cache["content_hash"]:
        raise ValueError("eligibility and convergence cache conflict")
    dates = tuple(date.fromisoformat(value) for value in cache["trade_dates"])
    scores = np.load(args.l3_scores, mmap_mode="r")
    prices = np.load(args.market_matrices / "adjusted_open.npy", mmap_mode="r")
    flags = np.load(args.eligibility_dir / "eligibility_flags.npy", mmap_mode="r")
    if scores.shape != prices.shape or scores.shape != flags.shape:
        raise ValueError("universe Shapley inputs do not align")
    benchmark = _benchmark_returns(args.benchmark_proxy)
    rows: list[dict[str, Any]] = []
    for fold in l3["folds"]:
        values = _subset_values(
            scores=scores,
            prices=prices,
            flags=flags,
            dates=dates,
            folds=[fold],
            benchmark=benchmark,
            target_count=int(fold["l4_selection"]["selected"]["target_count"]),
            frequency=RebalanceFrequency(fold["l4_selection"]["selected"]["frequency"]),
        )
        rows.extend(_shapley_rows(str(fold["fold"]), values))
    aggregate_values = _subset_values(
        scores=scores,
        prices=prices,
        flags=flags,
        dates=dates,
        folds=l3["folds"],
        benchmark=benchmark,
        target_count=None,
        frequency=None,
    )
    rows.extend(_shapley_rows("ALL", aggregate_values))
    exposure_rows = _topk_exposures(
        scores=scores,
        prices=prices,
        flags=flags,
        dates=dates,
        folds=l3["folds"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = _csv_text(rows)
    exposure_text = _csv_text(exposure_rows)
    _write(args.output_dir / "universe_filter_shapley.csv", text)
    _write(args.output_dir / "topk_eligibility_exposure.csv", exposure_text)
    aggregate = [row for row in rows if row["fold"] == "ALL"]
    stable = {
        "status": "PASS_RESEARCH_ONLY",
        "stage": "UNIVERSE_FILTER_EXACT_SHAPLEY",
        "diagnostic_only": True,
        "outer_oos_used_for_retuning": False,
        "aggregate": aggregate,
        "value_function": "annual excess of fold-selected ideal adjusted-open portfolio",
        "subset_count": 2 ** len(_FEATURES),
        "shapley_efficiency_error": abs(
            sum(float(row["shapley_annual_excess_effect"]) for row in aggregate)
            - (aggregate_values[31] - aggregate_values[0])
        ),
        "csv_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "topk_exposure_sha256": hashlib.sha256(exposure_text.encode()).hexdigest(),
        "provenance": {
            "cache_content_hash": cache["content_hash"],
            "l3_content_hash": l3["content_hash"],
            "eligibility_content_hash": eligibility["content_hash"],
            "code_version": args.code_version,
        },
    }
    stable["content_hash"] = _hash(stable)
    payload = {**stable, "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
    _write(args.output_dir / "universe_filter_shapley.json", json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact Shapley attribution of L4 universe filters")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--market-matrices", type=Path, required=True)
    parser.add_argument("--eligibility-dir", type=Path, required=True)
    parser.add_argument("--l3-scores", type=Path, required=True)
    parser.add_argument("--l3-metadata", type=Path, required=True)
    parser.add_argument("--benchmark-proxy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    return parser.parse_args()


def _subset_values(
    *,
    scores: np.ndarray[Any, Any],
    prices: np.ndarray[Any, Any],
    flags: np.ndarray[Any, Any],
    dates: tuple[date, ...],
    folds: list[dict[str, Any]],
    benchmark: dict[date, float],
    target_count: int | None,
    frequency: RebalanceFrequency | None,
) -> dict[int, float]:
    paths: dict[int, list[float]] = {subset: [] for subset in range(32)}
    benchmarks: list[float] = []
    for fold in folds:
        selected = fold["l4_selection"]["selected"]
        count = target_count or int(selected["target_count"])
        cadence = frequency or RebalanceFrequency(selected["frequency"])
        for subset in range(32):
            mask = None if subset == 0 else (np.asarray(flags) & subset) == subset
            daily = _simulate_control(
                scores=scores,
                prices=prices,
                dates=dates,
                folds=[fold],
                benchmark=benchmark,
                target_count=count,
                frequency=cadence,
                cost_bps=0.0,
                eligibility_mask=mask,
            )
            paths[subset].extend(float(item["strategy_return"]) for item in daily)
            if subset == 0:
                benchmarks.extend(float(item["benchmark_return"]) for item in daily)
    benchmark_annual = _annualized(benchmarks)
    return {subset: _annualized(path) - benchmark_annual for subset, path in paths.items()}


def _shapley_rows(fold: str, values: dict[int, float]) -> list[dict[str, Any]]:
    count = len(_FEATURES)
    factorial = math.factorial
    rows: list[dict[str, Any]] = []
    for name, bit in _FEATURES:
        contribution = 0.0
        for subset in range(2**count):
            if subset & bit:
                continue
            size = subset.bit_count()
            weight = factorial(size) * factorial(count - size - 1) / factorial(count)
            contribution += weight * (values[subset | bit] - values[subset])
        rows.append(
            {
                "fold": fold,
                "filter": name,
                "shapley_annual_excess_effect": contribution,
                "unfiltered_annual_excess": values[0],
                "fully_filtered_annual_excess": values[31],
                "total_filter_effect": values[31] - values[0],
            }
        )
    return rows


def _topk_exposures(
    *,
    scores: np.ndarray[Any, Any],
    prices: np.ndarray[Any, Any],
    flags: np.ndarray[Any, Any],
    dates: tuple[date, ...],
    folds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    date_index = {value: index for index, value in enumerate(dates)}
    rows: list[dict[str, Any]] = []
    for fold in folds:
        selected = fold["l4_selection"]["selected"]
        target_count = int(selected["target_count"])
        frequency = RebalanceFrequency(selected["frequency"])
        start = date.fromisoformat(fold["test_start"])
        end = date.fromisoformat(fold["test_end"])
        sessions = tuple(value for value in dates if start <= value <= end)
        for trade_date in generate_rebalance_dates(sessions, frequency):
            position = date_index[trade_date]
            score = np.asarray(scores[position], dtype=float)
            eligible = np.flatnonzero(np.isfinite(score) & np.isfinite(prices[position]))
            top = eligible[np.argsort(score[eligible], kind="stable")[-target_count:]]
            selected_flags = np.asarray(flags[position, top], dtype=np.uint8)
            row: dict[str, Any] = {
                "fold": int(fold["fold"]),
                "trade_date": trade_date.isoformat(),
                "target_count": target_count,
                "frequency": frequency.value,
                "selected_count": len(top),
                "fully_eligible_count": int(np.sum(selected_flags == 31)),
            }
            for name, bit in _FEATURES:
                row[f"failed_{name.lower()}_count"] = int(np.sum((selected_flags & bit) == 0))
            rows.append(row)
    return rows


def _annualized(returns: list[float]) -> float:
    total = math.prod(1 + value for value in returns)
    return total ** (252 / len(returns)) - 1 if total > 0 else -1.0


def _benchmark_returns(path: Path) -> dict[date, float]:
    with path.open(newline="") as stream:
        return {
            date.fromisoformat(row["trade_date"]): float(row["return_rate"])
            for row in csv.DictReader(stream)
        }


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
