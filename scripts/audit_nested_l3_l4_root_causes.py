#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from aquant.factors.evaluation.institutional import (  # type: ignore[import-untyped]
    newey_west_mean_t,
)
from aquant.factors.operators.cross_sectional import cs_rank  # type: ignore[import-untyped]
from aquant.strategies.microcap.experiments import (  # type: ignore[import-untyped]
    RebalanceFrequency,
    generate_rebalance_dates,
)

FloatArray = npt.NDArray[np.float64]


def main() -> None:
    args = _arguments()
    cache = json.loads((args.cache_dir / "metadata.json").read_text())
    l3 = json.loads(args.l3_metadata.read_text())
    l4 = json.loads(args.l4_report.read_text())
    matrix_metadata = json.loads((args.market_matrices / "market_matrices.json").read_text())
    _validate_metadata(cache, l3, l4, matrix_metadata)
    dates = tuple(date.fromisoformat(value) for value in cache["trade_dates"])
    raw_close = np.load(args.cache_dir / "close.npy", mmap_mode="r")
    adjusted_close = np.load(args.market_matrices / "adjusted_close.npy", mmap_mode="r")
    adjusted_open = np.load(args.market_matrices / "adjusted_open.npy", mmap_mode="r")
    adjustment_factor = np.load(args.market_matrices / "adjustment_factor.npy", mmap_mode="r")
    scores = np.load(args.l3_scores, mmap_mode="r")
    if any(array.shape != scores.shape for array in (raw_close, adjusted_close, adjusted_open)):
        raise ValueError("scores and market matrices must align")

    raw_labels = _pit_labels(raw_close, 5)
    adjusted_close_labels = _pit_labels(adjusted_close, 5)
    adjusted_open_labels = _pit_labels(adjusted_open, 5)
    fold_by_position, target_by_position = _fold_arrays(dates, l4)
    oos_positions = np.flatnonzero(fold_by_position >= 0)
    benchmark = _benchmark_returns(args.benchmark_proxy)
    daily_rows, decile_rows = _daily_diagnostics(
        dates=dates,
        positions=oos_positions,
        scores=scores,
        raw_labels=raw_labels,
        adjusted_close_labels=adjusted_close_labels,
        adjusted_open_labels=adjusted_open_labels,
        adjustment_factor=adjustment_factor,
        fold_by_position=fold_by_position,
        target_by_position=target_by_position,
        benchmark=benchmark,
    )
    rolling_rows = _rolling_diagnostics(daily_rows, windows=(20, 60, 120))
    regime_rows = _regime_diagnostics(daily_rows, benchmark)
    control_rows, control_daily = _control_matrix(
        scores=scores,
        adjusted_close=adjusted_close,
        adjusted_open=adjusted_open,
        dates=dates,
        l3=l3,
        benchmark=benchmark,
    )
    hypotheses = _hypothesis_tests(daily_rows)
    main_effects = _factorial_effects(control_rows)
    summary = _root_cause_summary(
        daily_rows=daily_rows,
        regime_rows=regime_rows,
        control_rows=control_rows,
        hypotheses=hypotheses,
        main_effects=main_effects,
        l4=l4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "daily_signal_diagnostics.csv": _csv_text(daily_rows),
        "rolling_diagnostics.csv": _csv_text(rolling_rows),
        "decile_returns.csv": _csv_text(decile_rows),
        "regime_diagnostics.csv": _csv_text(regime_rows),
        "control_matrix.csv": _csv_text(control_rows),
        "control_daily_returns.csv": _csv_text(control_daily),
        "hypothesis_tests.csv": _csv_text(hypotheses),
        "factorial_effects.csv": _csv_text(main_effects),
    }
    evidence = {}
    for name, text in artifacts.items():
        _write(args.output_dir / name, text)
        evidence[name] = {"sha256": _text_hash(text), "rows": text.count("\n") - 1}
    stable: dict[str, Any] = {
        "status": "PASS_RESEARCH_ONLY",
        "stage": "NESTED_L3_L4_ROOT_CAUSE_AUDIT",
        "diagnostic_only": True,
        "outer_oos_used_for_retuning": False,
        "period": {
            "start": dates[int(oos_positions[0])].isoformat(),
            "end": dates[int(oos_positions[-1])].isoformat(),
            "sessions": len(oos_positions),
        },
        "label_definitions": {
            "raw_close_5d": "raw_close[T+6]/raw_close[T+1]-1",
            "adjusted_close_5d": "adjusted_close[T+6]/adjusted_close[T+1]-1",
            "adjusted_open_5d": "adjusted_open[T+6]/adjusted_open[T+1]-1",
        },
        "bootstrap": {"replicates": 2000, "block_sessions": 21, "seed": 20260826},
        "summary": summary,
        "artifacts": evidence,
        "provenance": {
            "cache_content_hash": cache["content_hash"],
            "l3_content_hash": l3["content_hash"],
            "l4_content_hash": l4["content_hash"],
            "market_matrix_content_hash": matrix_metadata["content_hash"],
            "code_version": args.code_version,
        },
        "limitations": [
            "The outer OOS has already been observed and is used for diagnosis only.",
            "Signal-only control portfolios omit ST, suspension, board-lot, and cash constraints.",
            "Control paths are not candidates for model selection or production admission.",
            "The benchmark is an internal PIT all-A equal-weight proxy, not official Wind.",
        ],
    }
    stable["content_hash"] = _hash(stable)
    payload = {**stable, "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
    _write(args.output_dir / "root_cause_summary.json", json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {"status": payload["status"], "content_hash": payload["content_hash"], **summary}
        )
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit nested L3-to-L4 return conversion")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--market-matrices", type=Path, required=True)
    parser.add_argument("--l3-scores", type=Path, required=True)
    parser.add_argument("--l3-metadata", type=Path, required=True)
    parser.add_argument("--l4-report", type=Path, required=True)
    parser.add_argument("--benchmark-proxy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    return parser.parse_args()


def _validate_metadata(
    cache: dict[str, Any],
    l3: dict[str, Any],
    l4: dict[str, Any],
    matrices: dict[str, Any],
) -> None:
    if cache.get("status") != "PASS" or matrices.get("status") != "PASS":
        raise ValueError("root-cause inputs must be complete")
    if matrices.get("cache_content_hash") != cache.get("content_hash"):
        raise ValueError("market matrices and convergence cache conflict")
    if l3.get("outer_test_used_for_model_selection") is not False:
        raise ValueError("L3 outer test isolation is not attested")
    if l4.get("outer_test_used_for_optimization") is not False:
        raise ValueError("L4 outer test isolation is not attested")


def _pit_labels(prices: npt.ArrayLike, horizon: int) -> FloatArray:
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 2 or horizon <= 0:
        raise ValueError("price labels require a matrix and positive horizon")
    result = np.full(values.shape, np.nan)
    if horizon + 1 < len(values):
        with np.errstate(all="ignore"):
            result[: -(horizon + 1)] = values[horizon + 1 :] / values[1:-horizon] - 1
    result[~np.isfinite(result)] = np.nan
    return result


def _fold_arrays(
    dates: tuple[date, ...], l4: dict[str, Any]
) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
    folds = np.full(len(dates), -1, dtype=np.int16)
    targets = np.full(len(dates), -1, dtype=np.int16)
    index = {value: position for position, value in enumerate(dates)}
    configs = {int(item["fold"]): item["selected"] for item in l4["fold_configurations"]}
    for fold in l4["fold_performance"]:
        number = int(fold["fold"])
        start = index[date.fromisoformat(fold["test_start"])]
        end = index[date.fromisoformat(fold["test_end"])] + 1
        folds[start:end] = number
        targets[start:end] = int(configs[number]["target_count"])
    return folds, targets


def _benchmark_returns(path: Path) -> dict[date, float]:
    rows = csv.DictReader(path.open())
    return {date.fromisoformat(row["trade_date"]): float(row["return_rate"]) for row in rows}


def _daily_diagnostics(
    *,
    dates: tuple[date, ...],
    positions: npt.NDArray[np.int64],
    scores: npt.ArrayLike,
    raw_labels: npt.ArrayLike,
    adjusted_close_labels: npt.ArrayLike,
    adjusted_open_labels: npt.ArrayLike,
    adjustment_factor: npt.ArrayLike,
    fold_by_position: npt.NDArray[np.int16],
    target_by_position: npt.NDArray[np.int16],
    benchmark: dict[date, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    score_values = np.asarray(scores)
    labels = {
        "raw_close": np.asarray(raw_labels),
        "adjusted_close": np.asarray(adjusted_close_labels),
        "adjusted_open": np.asarray(adjusted_open_labels),
    }
    factors = np.asarray(adjustment_factor)
    result: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    benchmark_dates = tuple(sorted(benchmark))
    benchmark_index = {value: index for index, value in enumerate(benchmark_dates)}
    benchmark_values = np.asarray([benchmark[value] for value in benchmark_dates])
    for position in positions:
        index = int(position)
        day = dates[index]
        row: dict[str, Any] = {
            "trade_date": day.isoformat(),
            "fold": int(fold_by_position[index]),
            "target_count": int(target_by_position[index]),
        }
        score = np.asarray(score_values[index], dtype=np.float64)
        adjusted = np.asarray(labels["adjusted_open"][index], dtype=np.float64)
        raw = np.asarray(labels["raw_close"][index], dtype=np.float64)
        adjusted_close = np.asarray(labels["adjusted_close"][index], dtype=np.float64)
        valid_adjusted = np.isfinite(score) & np.isfinite(adjusted)
        adjustment_event = (
            np.isfinite(raw) & np.isfinite(adjusted_close) & (np.abs(raw - adjusted_close) > 1e-7)
        )
        row["coverage"] = int(np.count_nonzero(valid_adjusted))
        row["adjustment_event_count"] = int(np.count_nonzero(adjustment_event))
        row["adjustment_event_share"] = (
            float(np.mean(adjustment_event[valid_adjusted])) if np.any(valid_adjusted) else math.nan
        )
        if index + 6 < len(factors):
            changed = (
                np.isfinite(factors[index + 1])
                & np.isfinite(factors[index + 6])
                & (np.abs(factors[index + 6] / factors[index + 1] - 1) > 1e-9)
            )
            row["adj_factor_change_count"] = int(np.count_nonzero(changed))
        else:
            row["adj_factor_change_count"] = 0
        for name, matrix in labels.items():
            metrics, deciles = _cross_section_metrics(
                score,
                np.asarray(matrix[index], dtype=np.float64),
                int(target_by_position[index]),
            )
            row.update({f"{name}_{key}": value for key, value in metrics.items()})
            if name == "adjusted_open":
                decile_rows.extend(
                    {
                        "trade_date": day.isoformat(),
                        "fold": int(fold_by_position[index]),
                        "decile": decile,
                        "mean_forward_return": value,
                    }
                    for decile, value in enumerate(deciles, start=1)
                )
        row["positive_ic_negative_topk"] = bool(
            row["adjusted_open_rank_ic"] > 0 and row["adjusted_open_topk_excess"] < 0
        )
        row["raw_minus_adjusted_rank_ic"] = row["raw_close_rank_ic"] - row["adjusted_close_rank_ic"]
        row["close_minus_open_rank_ic"] = (
            row["adjusted_close_rank_ic"] - row["adjusted_open_rank_ic"]
        )
        row["raw_minus_adjusted_topk_excess"] = (
            row["raw_close_topk_excess"] - row["adjusted_close_topk_excess"]
        )
        row["close_minus_open_topk_excess"] = (
            row["adjusted_close_topk_excess"] - row["adjusted_open_topk_excess"]
        )
        bpos = benchmark_index.get(day)
        row["benchmark_60d_return"] = (
            float(np.prod(1 + benchmark_values[bpos - 59 : bpos + 1]) - 1)
            if bpos is not None and bpos >= 59
            else math.nan
        )
        row["benchmark_20d_annual_volatility"] = (
            float(np.std(benchmark_values[bpos - 19 : bpos + 1], ddof=1) * np.sqrt(252))
            if bpos is not None and bpos >= 19
            else math.nan
        )
        result.append(row)
    return result, decile_rows


def _cross_section_metrics(
    score: FloatArray, returns: FloatArray, target_count: int
) -> tuple[dict[str, float | int], list[float]]:
    valid = np.isfinite(score) & np.isfinite(returns)
    count = int(np.count_nonzero(valid))
    empty = {
        "rank_ic": math.nan,
        "legacy_ordinal_rank_ic": math.nan,
        "pearson_ic": math.nan,
        "slope_per_score_sd": math.nan,
        "topk_return": math.nan,
        "universe_return": math.nan,
        "topk_excess": math.nan,
        "top10pct_excess": math.nan,
        "tail_rank_ic": math.nan,
        "long_short_decile": math.nan,
        "topk_hit_ratio": math.nan,
        "observations": count,
        "score_unique_ratio": math.nan,
        "score_largest_tie_share": math.nan,
    }
    if count < max(20, target_count):
        return empty, [math.nan] * 10
    values = score[valid]
    outcomes = returns[valid]
    order = np.argsort(values, kind="stable")
    selected = order[-min(target_count, count) :]
    decile_count = max(2, count // 10)
    top_decile = order[-decile_count:]
    bottom_decile = order[:decile_count]
    score_std = float(np.std(values, ddof=1))
    _, tie_counts = np.unique(values, return_counts=True)
    slope = float(np.cov(values, outcomes, ddof=1)[0, 1] / score_std) if score_std > 0 else math.nan
    universe = float(np.mean(outcomes))
    top_return = float(np.mean(outcomes[selected]))
    chunks = np.array_split(order, 10)
    deciles = [float(np.mean(outcomes[chunk])) if len(chunk) else math.nan for chunk in chunks]
    return (
        {
            "rank_ic": _rank_correlation(values, outcomes),
            "legacy_ordinal_rank_ic": _legacy_rank_correlation(values, outcomes),
            "pearson_ic": _correlation(values, outcomes),
            "slope_per_score_sd": slope,
            "topk_return": top_return,
            "universe_return": universe,
            "topk_excess": top_return - universe,
            "top10pct_excess": float(np.mean(outcomes[top_decile])) - universe,
            "tail_rank_ic": _rank_correlation(values[top_decile], outcomes[top_decile]),
            "long_short_decile": float(
                np.mean(outcomes[top_decile]) - np.mean(outcomes[bottom_decile])
            ),
            "topk_hit_ratio": float(np.mean(outcomes[selected] > universe)),
            "observations": count,
            "score_unique_ratio": float(len(tie_counts) / count),
            "score_largest_tie_share": float(np.max(tie_counts) / count),
        },
        deciles,
    )


def _rolling_diagnostics(
    rows: list[dict[str, Any]], windows: tuple[int, ...]
) -> list[dict[str, Any]]:
    metrics = (
        "adjusted_open_rank_ic",
        "adjusted_open_topk_excess",
        "adjusted_open_tail_rank_ic",
        "adjusted_open_slope_per_score_sd",
    )
    result = []
    for index, row in enumerate(rows):
        output: dict[str, Any] = {"trade_date": row["trade_date"], "fold": row["fold"]}
        for window in windows:
            start = max(0, index - window + 1)
            for metric in metrics:
                values = np.asarray([item[metric] for item in rows[start : index + 1]], dtype=float)
                finite = values[np.isfinite(values)]
                output[f"{metric}_{window}d"] = (
                    float(np.mean(finite)) if len(finite) >= max(5, window // 2) else math.nan
                )
        result.append(output)
    return result


def _regime_diagnostics(
    rows: list[dict[str, Any]], benchmark: dict[date, float]
) -> list[dict[str, Any]]:
    del benchmark
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        momentum = float(row["benchmark_60d_return"])
        volatility = float(row["benchmark_20d_annual_volatility"])
        trend = "UNCLASSIFIED"
        if np.isfinite(momentum):
            trend = "BULL" if momentum > 0.05 else "BEAR" if momentum < -0.05 else "SIDEWAYS"
        vol = (
            "UNCLASSIFIED"
            if not np.isfinite(volatility)
            else "HIGH_VOL"
            if volatility > 0.25
            else "LOW_VOL"
        )
        grouped[("ALL", trend, vol)].append(row)
        grouped[(str(row["fold"]), trend, vol)].append(row)
    result = []
    for (fold, trend, vol), members in sorted(grouped.items()):
        if len(members) < 10:
            continue
        result.append(
            {
                "fold": fold,
                "trend_regime": trend,
                "volatility_regime": vol,
                "sessions": len(members),
                "mean_rank_ic": _finite_mean(members, "adjusted_open_rank_ic"),
                "mean_tail_rank_ic": _finite_mean(members, "adjusted_open_tail_rank_ic"),
                "mean_topk_excess": _finite_mean(members, "adjusted_open_topk_excess"),
                "positive_ic_negative_topk_ratio": float(
                    np.mean([bool(item["positive_ic_negative_topk"]) for item in members])
                ),
                "mean_5d_universe_return": _finite_mean(members, "adjusted_open_universe_return"),
            }
        )
    return result


def _control_matrix(
    *,
    scores: npt.ArrayLike,
    adjusted_close: npt.ArrayLike,
    adjusted_open: npt.ArrayLike,
    dates: tuple[date, ...],
    l3: dict[str, Any],
    benchmark: dict[date, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    score_values = np.asarray(scores)
    prices = {
        "adjusted_close": np.asarray(adjusted_close),
        "adjusted_open": np.asarray(adjusted_open),
    }
    rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for target_count in (20, 30, 50, 80, 100):
        for frequency in (RebalanceFrequency.WEEKLY, RebalanceFrequency.MONTHLY):
            for basis, matrix in prices.items():
                for cost_bps in (0.0, 10.0):
                    daily = _simulate_control(
                        scores=score_values,
                        prices=matrix,
                        dates=dates,
                        folds=l3["folds"],
                        benchmark=benchmark,
                        target_count=target_count,
                        frequency=frequency,
                        cost_bps=cost_bps,
                    )
                    strategy = np.asarray([item["strategy_return"] for item in daily])
                    benchmark_returns = np.asarray([item["benchmark_return"] for item in daily])
                    excess = strategy - benchmark_returns
                    rows.append(
                        {
                            "target_count": target_count,
                            "frequency": frequency.value,
                            "price_basis": basis,
                            "cost_bps": cost_bps,
                            "sessions": len(daily),
                            "annual_return": _annualized(strategy),
                            "annual_benchmark_return": _annualized(benchmark_returns),
                            "annual_excess_return": _annualized(strategy)
                            - _annualized(benchmark_returns),
                            "excess_sharpe": _sharpe(excess),
                            "annual_turnover": float(np.sum([item["turnover"] for item in daily]))
                            * 252
                            / len(daily),
                            "maximum_drawdown": _maximum_drawdown(strategy),
                        }
                    )
                    daily_rows.extend(
                        {
                            "target_count": target_count,
                            "frequency": frequency.value,
                            "price_basis": basis,
                            "cost_bps": cost_bps,
                            **item,
                        }
                        for item in daily
                    )
    return rows, daily_rows


def _simulate_control(
    *,
    scores: npt.NDArray[np.generic],
    prices: npt.NDArray[np.generic],
    dates: tuple[date, ...],
    folds: list[dict[str, Any]],
    benchmark: dict[date, float],
    target_count: int,
    frequency: RebalanceFrequency,
    cost_bps: float,
    eligibility_mask: npt.NDArray[np.generic] | None = None,
) -> list[dict[str, Any]]:
    date_index = {value: index for index, value in enumerate(dates)}
    result: list[dict[str, Any]] = []
    for fold in folds:
        start = date.fromisoformat(fold["test_start"])
        end = date.fromisoformat(fold["test_end"])
        fold_dates = tuple(value for value in dates if start <= value <= end)
        rebalance_dates = set(generate_rebalance_dates(fold_dates, frequency))
        active: FloatArray = np.zeros(prices.shape[1], dtype=np.float64)
        pending: FloatArray | None = None
        for day in fold_dates:
            index = date_index[day]
            if index == 0:
                continue
            with np.errstate(all="ignore"):
                asset_returns = (
                    np.asarray(prices[index], dtype=float)
                    / np.asarray(prices[index - 1], dtype=float)
                    - 1
                )
            valid = np.isfinite(asset_returns)
            strategy_return = float(np.sum(active[valid] * asset_returns[valid]))
            turnover = 0.0
            if pending is not None:
                turnover = float(np.sum(np.abs(pending - active)))
                strategy_return -= turnover * cost_bps / 10_000
                active = pending
                pending = None
            if day in rebalance_dates:
                row = np.asarray(scores[index], dtype=float)
                eligible_values = np.isfinite(row) & np.isfinite(prices[index])
                if eligibility_mask is not None:
                    eligible_values &= np.asarray(eligibility_mask[index], dtype=bool)
                eligible = np.flatnonzero(eligible_values)
                selected = eligible[np.argsort(row[eligible], kind="stable")[-target_count:]]
                pending = np.zeros_like(active)
                if len(selected):
                    pending[selected] = 1 / len(selected)
            result.append(
                {
                    "trade_date": day.isoformat(),
                    "fold": int(fold["fold"]),
                    "strategy_return": strategy_return,
                    "benchmark_return": benchmark.get(day, 0.0),
                    "turnover": turnover,
                }
            )
    return result


def _hypothesis_tests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions: dict[str, npt.NDArray[np.float64]] = {
        "H1_ADJUSTED_OPEN_RANK_IC_POSITIVE": np.asarray(
            [row["adjusted_open_rank_ic"] for row in rows], dtype=float
        ),
        "H2_ADJUSTED_OPEN_TOPK_EXCESS_POSITIVE": np.asarray(
            [row["adjusted_open_topk_excess"] for row in rows], dtype=float
        ),
        "H3_RAW_LABEL_RANKIC_BIAS": np.asarray(
            [row["raw_minus_adjusted_rank_ic"] for row in rows], dtype=float
        ),
        "H4_RAW_LABEL_TOPK_BIAS": np.asarray(
            [row["raw_minus_adjusted_topk_excess"] for row in rows], dtype=float
        ),
        "H5_CLOSE_OPEN_RANKIC_MISMATCH": np.asarray(
            [row["close_minus_open_rank_ic"] for row in rows], dtype=float
        ),
        "H6_CLOSE_OPEN_TOPK_MISMATCH": np.asarray(
            [row["close_minus_open_topk_excess"] for row in rows], dtype=float
        ),
        "H7_TAIL_RANK_IC_POSITIVE": np.asarray(
            [row["adjusted_open_tail_rank_ic"] for row in rows], dtype=float
        ),
        "H8_LEGACY_ORDINAL_RANKIC_BIAS": np.asarray(
            [
                row["adjusted_open_legacy_ordinal_rank_ic"] - row["adjusted_open_rank_ic"]
                for row in rows
            ],
            dtype=float,
        ),
    }
    for fold in sorted({int(row["fold"]) for row in rows}):
        members = [row for row in rows if int(row["fold"]) == fold]
        definitions[f"FOLD_{fold}_RANK_IC"] = np.asarray(
            [row["adjusted_open_rank_ic"] for row in members], dtype=float
        )
        definitions[f"FOLD_{fold}_TOPK_EXCESS"] = np.asarray(
            [row["adjusted_open_topk_excess"] for row in members], dtype=float
        )
        definitions[f"FOLD_{fold}_LONG_SHORT_DECILE"] = np.asarray(
            [row["adjusted_open_long_short_decile"] for row in members], dtype=float
        )
    output: list[dict[str, Any]] = []
    for offset, (hypothesis, values) in enumerate(definitions.items()):
        finite = values[np.isfinite(values)]
        low, high, p_value = _block_bootstrap_mean(
            finite, block_size=21, replicates=2000, seed=20260826 + offset
        )
        output.append(
            {
                "hypothesis": hypothesis,
                "observations": len(finite),
                "mean": float(np.mean(finite)),
                "newey_west_t": newey_west_mean_t(finite, max_lag=10),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_two_sided_p": p_value,
            }
        )
    q_values = _benjamini_hochberg([float(item["bootstrap_two_sided_p"]) for item in output])
    for item, q_value in zip(output, q_values, strict=True):
        item["fdr_q_value"] = q_value
        item["fdr_significant_5pct"] = q_value <= 0.05
    return output


def _factorial_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def mean(where: Any) -> float:
        values = [float(row["annual_excess_return"]) for row in rows if where(row)]
        return float(np.mean(values))

    effects = [
        {
            "effect": "OPEN_MINUS_CLOSE",
            "annual_excess_effect": mean(lambda row: row["price_basis"] == "adjusted_open")
            - mean(lambda row: row["price_basis"] == "adjusted_close"),
        },
        {
            "effect": "10BPS_MINUS_ZERO_COST",
            "annual_excess_effect": mean(lambda row: float(row["cost_bps"]) == 10.0)
            - mean(lambda row: float(row["cost_bps"]) == 0.0),
        },
        {
            "effect": "WEEKLY_MINUS_MONTHLY",
            "annual_excess_effect": mean(lambda row: row["frequency"] == "weekly")
            - mean(lambda row: row["frequency"] == "monthly"),
        },
    ]
    for count in (20, 30, 50, 80, 100):
        effects.append(
            {
                "effect": f"TARGET_{count}_CENTERED",
                "annual_excess_effect": mean(
                    lambda row, count=count: int(row["target_count"]) == count
                )
                - mean(lambda _row: True),
            }
        )
    weekly_cost = mean(
        lambda row: row["frequency"] == "weekly" and float(row["cost_bps"]) == 10.0
    ) - mean(lambda row: row["frequency"] == "weekly" and float(row["cost_bps"]) == 0.0)
    monthly_cost = mean(
        lambda row: row["frequency"] == "monthly" and float(row["cost_bps"]) == 10.0
    ) - mean(lambda row: row["frequency"] == "monthly" and float(row["cost_bps"]) == 0.0)
    effects.append(
        {
            "effect": "FREQUENCY_X_COST_INTERACTION",
            "annual_excess_effect": weekly_cost - monthly_cost,
        }
    )
    return effects


def _root_cause_summary(
    *,
    daily_rows: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    main_effects: list[dict[str, Any]],
    l4: dict[str, Any],
) -> dict[str, Any]:
    hypothesis = {item["hypothesis"]: item for item in hypotheses}
    effects = {item["effect"]: item["annual_excess_effect"] for item in main_effects}
    positive_ic_negative_topk = float(
        np.mean([bool(item["positive_ic_negative_topk"]) for item in daily_rows])
    )
    event_days = [item for item in daily_rows if int(item["adj_factor_change_count"]) > 0]
    best = max(control_rows, key=lambda item: float(item["annual_excess_return"]))
    aggregate_regimes = [
        item
        for item in regime_rows
        if item["fold"] == "ALL" and item["trend_regime"] != "UNCLASSIFIED"
    ]
    worst_regime = min(aggregate_regimes, key=lambda item: float(item["mean_topk_excess"]))
    return {
        "adjusted_open_mean_rank_ic": hypothesis["H1_ADJUSTED_OPEN_RANK_IC_POSITIVE"]["mean"],
        "adjusted_open_mean_topk_5d_excess": hypothesis["H2_ADJUSTED_OPEN_TOPK_EXCESS_POSITIVE"][
            "mean"
        ],
        "adjusted_open_mean_tail_rank_ic": hypothesis["H7_TAIL_RANK_IC_POSITIVE"]["mean"],
        "legacy_ordinal_rankic_bias": hypothesis["H8_LEGACY_ORDINAL_RANKIC_BIAS"]["mean"],
        "mean_score_unique_ratio": _finite_mean(daily_rows, "adjusted_open_score_unique_ratio"),
        "mean_largest_tie_share": _finite_mean(daily_rows, "adjusted_open_score_largest_tie_share"),
        "positive_ic_negative_topk_day_ratio": positive_ic_negative_topk,
        "raw_label_rankic_bias": hypothesis["H3_RAW_LABEL_RANKIC_BIAS"]["mean"],
        "raw_label_topk_bias": hypothesis["H4_RAW_LABEL_TOPK_BIAS"]["mean"],
        "close_open_rankic_mismatch": hypothesis["H5_CLOSE_OPEN_RANKIC_MISMATCH"]["mean"],
        "close_open_topk_mismatch": hypothesis["H6_CLOSE_OPEN_TOPK_MISMATCH"]["mean"],
        "corporate_action_event_days": len(event_days),
        "open_minus_close_annual_excess_effect": effects["OPEN_MINUS_CLOSE"],
        "ten_bps_annual_excess_effect": effects["10BPS_MINUS_ZERO_COST"],
        "weekly_minus_monthly_annual_excess_effect": effects["WEEKLY_MINUS_MONTHLY"],
        "frequency_cost_interaction": effects["FREQUENCY_X_COST_INTERACTION"],
        "best_diagnostic_control": best,
        "worst_regime": worst_regime,
        "actual_l4_annual_excess_return": l4["performance"]["annual_excess_return"],
    }


def _rank_correlation(left: FloatArray, right: FloatArray) -> float:
    if len(left) < 2:
        return math.nan
    return _correlation(
        np.asarray(cs_rank(left), dtype=float), np.asarray(cs_rank(right), dtype=float)
    )


def _legacy_ordinal_rank(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values))
    return ranks


def _legacy_rank_correlation(left: FloatArray, right: FloatArray) -> float:
    return _correlation(_legacy_ordinal_rank(left), _legacy_ordinal_rank(right))


def _correlation(left: FloatArray, right: FloatArray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _finite_mean(rows: list[dict[str, Any]], field: str) -> float:
    values = np.asarray([row[field] for row in rows], dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else math.nan


def _block_bootstrap_mean(
    values: FloatArray, *, block_size: int, replicates: int, seed: int
) -> tuple[float, float, float]:
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) < block_size or replicates < 100:
        raise ValueError("block bootstrap sample or replicate count is insufficient")
    generator = np.random.default_rng(seed)
    blocks = math.ceil(len(sample) / block_size)
    maximum_start = len(sample) - block_size
    means = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        starts = generator.integers(0, maximum_start + 1, size=blocks)
        resampled = np.concatenate([sample[start : start + block_size] for start in starts])[
            : len(sample)
        ]
        means[replicate] = float(np.mean(resampled))
    low, high = np.quantile(means, (0.025, 0.975))
    p_value = min(1.0, 2 * min(float(np.mean(means <= 0)), float(np.mean(means >= 0))))
    return float(low), float(high), p_value


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = count - reverse_rank + 1
        running = min(running, p_values[int(index)] * count / rank)
        adjusted[int(index)] = running
    return adjusted.tolist()


def _annualized(returns: npt.ArrayLike) -> float:
    values = np.asarray(returns, dtype=float)
    total = float(np.prod(1 + values))
    return total ** (252 / len(values)) - 1 if len(values) and total > 0 else -1.0


def _sharpe(returns: npt.ArrayLike) -> float:
    values = np.asarray(returns, dtype=float)
    standard_deviation = float(np.std(values, ddof=1))
    return (
        float(np.mean(values) / standard_deviation * np.sqrt(252))
        if standard_deviation > 0
        else 0.0
    )


def _maximum_drawdown(returns: npt.ArrayLike) -> float:
    equity = np.cumprod(1 + np.asarray(returns, dtype=float))
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))
    return float(np.min(np.concatenate(([1.0], equity)) / peaks - 1))


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("CSV output must not be empty")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


if __name__ == "__main__":
    main()
