#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import statistics
from bisect import bisect_right
from datetime import date
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import yaml
from debug_wind_microcap_replication import _official_benchmark
from run_microcap_five_year_matrix import (
    DEFAULT_CAPITAL,
    DEFAULT_END,
    DEFAULT_RELEASE,
    DEFAULT_START,
    _date,
    _load_targets,
)

from aquant.backtest import EventDrivenBacktest, UnfilledOrderPolicy
from aquant.data.history import DuckDBMicrocapHistory, HistoryReleaseReader
from aquant.strategies.microcap import (
    CostScenario,
    MicrocapPrototype,
    MicrocapRankBandStrategy,
    RebalanceFrequency,
    generate_rebalance_dates,
    matcher_for_cost_scenario,
)

DEFAULT_WIND_CONFIG = Path("config/index_replication/wind_microcap_daily_equal_2024_2025.yaml")


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _write_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _drawdowns(values: list[float]) -> list[float]:
    peak = values[0]
    output: list[float] = []
    for value in values:
        peak = max(peak, value)
        output.append(value / peak - 1.0)
    return output


def _returns(values: list[float]) -> list[float]:
    return [current / previous - 1.0 for previous, current in pairwise(values)]


def _annualized_sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        raise ValueError("Sharpe ratio requires at least two returns")
    volatility = statistics.stdev(returns)
    if volatility == 0:
        raise ValueError("Sharpe ratio is undefined for zero volatility")
    return statistics.mean(returns) / volatility * math.sqrt(52.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare executable-95 weekly base-cost with the Wind micro-cap index"
    )
    parser.add_argument("--release-directory", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--start-date", type=_date, default=DEFAULT_START)
    parser.add_argument("--end-date", type=_date, default=DEFAULT_END)
    parser.add_argument("--initial-capital", type=Decimal, default=DEFAULT_CAPITAL)
    parser.add_argument("--wind-config", type=Path, default=DEFAULT_WIND_CONFIG)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/executable95_wind_comparison"),
    )
    args = parser.parse_args()

    release = HistoryReleaseReader(args.release_directory)
    with DuckDBMicrocapHistory(args.release_directory) as history:
        trading_dates = history.trading_dates(args.start_date, args.end_date)
    targets, target_hash = _load_targets(
        args.release_directory,
        start_date=args.start_date,
        end_date=args.end_date,
        trading_dates=trading_dates,
    )
    executable_targets = targets[MicrocapPrototype.EXECUTABLE_95]
    ts_codes = tuple(
        sorted(
            {
                symbol.canonical.replace(".XSHG", ".SH").replace(".XSHE", ".SZ")
                for target in executable_targets.values()
                for symbol in target.symbols
            }
        )
    )
    with DuckDBMicrocapHistory(args.release_directory) as history:
        sessions = history.sessions(
            args.start_date,
            args.end_date,
            ts_codes=ts_codes,
        )
    signal_dates = generate_rebalance_dates(
        tuple(session.trade_date for session in sessions),
        RebalanceFrequency.WEEKLY,
    )
    strategy = MicrocapRankBandStrategy(
        targets={day: executable_targets[day] for day in signal_dates}
    )
    result = EventDrivenBacktest(
        run_id="executable-95-weekly-base-cost-wind-comparison",
        initial_cash=args.initial_capital,
        matcher=matcher_for_cost_scenario(CostScenario.BASE_COST),
        unfilled_order_policy=UnfilledOrderPolicy.CANCEL_AFTER_OPEN,
    ).run(sessions, strategy)

    config = yaml.safe_load(args.wind_config.read_text(encoding="utf-8"))
    index_code = str(config["index_code"])
    wind_weekly, _recent, _annual, benchmark_hash = _official_benchmark(
        config,
        index_code=index_code,
    )
    equity_by_date = {
        snapshot.asof_time.date(): float(snapshot.equity) for snapshot in result.equity_curve
    }
    strategy_dates = sorted(equity_by_date)
    aligned: list[tuple[date, date, float, float]] = []
    for benchmark_date, close in sorted(wind_weekly.items()):
        if not args.start_date <= benchmark_date <= args.end_date:
            continue
        position = bisect_right(strategy_dates, benchmark_date) - 1
        if position < 0:
            continue
        strategy_date = strategy_dates[position]
        if (benchmark_date - strategy_date).days > 7:
            continue
        aligned.append(
            (
                benchmark_date,
                strategy_date,
                equity_by_date[strategy_date],
                float(close),
            )
        )
    if len(aligned) < 52:
        raise ValueError("Wind comparison has insufficient aligned weekly observations")

    initial_strategy = aligned[0][2]
    initial_benchmark = aligned[0][3]
    strategy_nav = [row[2] / initial_strategy for row in aligned]
    benchmark_nav = [row[3] / initial_benchmark for row in aligned]
    excess_nav = [
        strategy_value / benchmark_value
        for strategy_value, benchmark_value in zip(
            strategy_nav,
            benchmark_nav,
            strict=True,
        )
    ]
    strategy_returns = _returns(strategy_nav)
    benchmark_returns = _returns(benchmark_nav)
    excess_returns = [
        (1.0 + strategy_return) / (1.0 + benchmark_return) - 1.0
        for strategy_return, benchmark_return in zip(
            strategy_returns,
            benchmark_returns,
            strict=True,
        )
    ]
    strategy_drawdown = _drawdowns(strategy_nav)
    benchmark_drawdown = _drawdowns(benchmark_nav)
    excess_drawdown = _drawdowns(excess_nav)
    combined_points = zip(
        aligned,
        strategy_nav,
        benchmark_nav,
        excess_nav,
        strategy_drawdown,
        benchmark_drawdown,
        excess_drawdown,
        strict=True,
    )
    points: list[dict[str, object]] = []
    for aligned_row in combined_points:
        (
            (
                benchmark_date,
                strategy_date,
                _equity,
                _close,
            ),
            strategy_value,
            benchmark_value,
            excess_value,
            strategy_dd,
            benchmark_dd,
            excess_dd,
        ) = aligned_row
        points.append(
            {
                "benchmark_date": benchmark_date.isoformat(),
                "strategy_date": strategy_date.isoformat(),
                "strategy_nav": round(strategy_value, 8),
                "benchmark_nav": round(benchmark_value, 8),
                "excess_nav": round(excess_value, 8),
                "strategy_drawdown": round(strategy_dd, 8),
                "benchmark_drawdown": round(benchmark_dd, 8),
                "excess_drawdown": round(excess_dd, 8),
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "strategy": "executable-95",
        "frequency": "weekly",
        "cost_scenario": "base-cost",
        "benchmark": index_code,
        "start_date": aligned[0][0].isoformat(),
        "end_date": aligned[-1][0].isoformat(),
        "weekly_observations": len(points),
        "strategy_total_return": strategy_nav[-1] - 1.0,
        "benchmark_total_return": benchmark_nav[-1] - 1.0,
        "geometric_excess_return": excess_nav[-1] - 1.0,
        "strategy_sharpe": _annualized_sharpe(strategy_returns),
        "excess_sharpe": _annualized_sharpe(excess_returns),
        "excess_weekly_win_rate": sum(value > 0 for value in excess_returns) / len(excess_returns),
        "strategy_max_drawdown": min(strategy_drawdown),
        "benchmark_max_drawdown": min(benchmark_drawdown),
        "excess_max_drawdown": min(excess_drawdown),
        "data_release_id": release.manifest.release_id,
        "target_hash": target_hash,
        "benchmark_source_hash": benchmark_hash,
        "strategy_final_state_hash": result.final_state_hash,
        "points": points,
    }
    payload["content_hash"] = _hash(payload)
    stem = f"executable95-vs-wind-{str(payload['content_hash'])[:12]}"
    output = args.output_directory / f"{stem}.json"
    _write_atomic(output, json.dumps(payload, ensure_ascii=False, indent=2))
    print(output)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "weekly_observations",
                    "strategy_total_return",
                    "benchmark_total_return",
                    "geometric_excess_return",
                    "strategy_sharpe",
                    "excess_sharpe",
                    "excess_weekly_win_rate",
                    "strategy_max_drawdown",
                    "benchmark_max_drawdown",
                    "excess_max_drawdown",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
