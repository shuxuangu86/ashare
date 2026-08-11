#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from aquant.backtest import EventDrivenBacktest, UnfilledOrderPolicy, calculate_metrics
from aquant.data.history import DuckDBMicrocapHistory
from aquant.strategies import (
    CostScenario,
    RebalanceFrequency,
    SingleFactorConfig,
    SingleFactorEqualWeightStrategy,
    SingleFactorObservation,
    SingleFactorSelector,
    SingleFactorSnapshot,
    generate_rebalance_dates,
    matcher_for_cost_scenario,
)
from aquant.strategies.all_a_equal_proxy import build_all_a_equal_weight_proxy

_BASE_COST_ASSUMPTIONS: dict[str, Any] = {
    "signal_time": "T_CLOSE",
    "execution_time": "T_PLUS_1_OPEN",
    "slippage_bps_each_fill": 5,
    "commission_rate": 0.0003,
    "minimum_commission_cny": 5,
    "sell_stamp_duty_rate_before_2023_08_28": 0.001,
    "sell_stamp_duty_rate_from_2023_08_28": 0.0005,
    "transfer_fee_rate_before_2022_04_29": 0.00002,
    "transfer_fee_rate_from_2022_04_29": 0.00001,
    "lot_size": 100,
    "maximum_prior_20d_average_volume_participation": 0.005,
    "security_status_required": True,
    "unfilled_order_policy": "CANCEL_AFTER_OPEN",
    "tradability_rules": "SUSPENSION_AND_PRICE_LIMIT_AWARE",
}


def main() -> None:
    args = _arguments()
    l3 = json.loads(args.l3_metadata.read_text())
    expected_l3_hash = l3.pop("content_hash")
    if expected_l3_hash != _hash(l3):
        raise ValueError("L3 metadata content hash mismatch")
    l3["content_hash"] = expected_l3_hash
    if l3.get("status") != "PASS" or l3.get("outer_test_used_for_model_selection") is not False:
        raise ValueError("L4 requires completed leakage-safe nested L3 metadata")
    if _file_hash(args.l3_scores) != l3["score_file_hash"]:
        raise ValueError("L3 score file hash mismatch")
    cache = json.loads((args.cache_dir / "metadata.json").read_text())
    if cache.get("status") != "PASS":
        raise ValueError("nested union cache is incomplete")
    dates = tuple(date.fromisoformat(value) for value in cache["trade_dates"])
    codes = tuple(str(value) for value in cache["ts_codes"])
    scores = np.load(args.l3_scores, mmap_mode="r")
    if scores.shape != (len(dates), len(codes)):
        raise ValueError("L3 scores do not align with nested union cache")
    start_date = date.fromisoformat(l3["folds"][0]["test_start"])
    end_date = date.fromisoformat(l3["folds"][-1]["test_end"])
    date_index = {value: index for index, value in enumerate(dates)}
    code_index = {value: index for index, value in enumerate(codes)}

    expected_session_dates = tuple(value for value in dates if start_date <= value <= end_date)
    with DuckDBMicrocapHistory(args.history_release) as history:
        session_dates = history.trading_dates(start_date, end_date)
        if session_dates != expected_session_dates:
            raise ValueError("history sessions and L3 trading dates differ")
        rebalance_configs = _rebalance_configs(l3, session_dates)
        # Build one PIT selection snapshot at a time. Only securities that can
        # actually be held need full event-engine bars, which avoids loading a
        # five-year all-market session cube into memory.
        snapshots: dict[date, SingleFactorSnapshot] = {}
        selected_codes: set[str] = set()
        for trade_date, config in rebalance_configs.items():
            risk = history.snapshot(trade_date)
            snapshot = _snapshots(
                trading_dates=(trade_date,),
                risk_snapshots={trade_date: risk},
                scores=scores,
                date_index=date_index,
                code_index=code_index,
            )[trade_date]
            snapshots[trade_date] = snapshot
            selected_codes.update(
                symbol.canonical.replace(".XSHG", ".SH").replace(".XSHE", ".SZ")
                for symbol in SingleFactorSelector(config).select(snapshot).symbols
            )
        sessions = history.sessions(
            start_date,
            end_date,
            ts_codes=tuple(sorted(selected_codes)),
        )
        release_id = history.release.manifest.release_id
    if tuple(session.trade_date for session in sessions) != expected_session_dates:
        raise ValueError("history sessions and L3 trading dates differ")
    initial_equity = Decimal("1000000")
    result = EventDrivenBacktest(
        run_id=f"nested-l3-l4-{expected_l3_hash[:12]}",
        initial_cash=initial_equity,
        matcher=matcher_for_cost_scenario(CostScenario.BASE_COST),
        unfilled_order_policy=UnfilledOrderPolicy.CANCEL_AFTER_OPEN,
    ).run(
        sessions,
        SingleFactorEqualWeightStrategy(
            snapshots=snapshots,
            rebalance_dates=set(rebalance_configs),
            config_by_rebalance_date=rebalance_configs,
        ),
    )
    metrics = calculate_metrics(result, initial_equity=initial_equity)
    orders = {order.order_id: order for order in result.orders}
    t_plus_one = bool(result.fills) and all(
        fill.occurred_at.date() > orders[fill.order_id].submitted_at.date() for fill in result.fills
    )
    if not t_plus_one:
        raise ValueError("nested L4 produced no fills or violated T+1 execution")
    proxy, proxy_metadata = build_all_a_equal_weight_proxy(
        args.history_release, start_date=start_date, end_date=end_date
    )
    proxy_returns = {item.trade_date: item.return_rate for item in proxy}
    strategy_returns = _strategy_returns(result.equity_curve, initial_equity)
    aligned_dates = tuple(
        day for day in session_dates if day in strategy_returns and day in proxy_returns
    )
    strategy = np.asarray([strategy_returns[day] for day in aligned_dates])
    benchmark = np.asarray([proxy_returns[day] for day in aligned_dates])
    performance = _performance(strategy, benchmark)
    fold_performance = _fold_performance(
        folds=l3["folds"],
        aligned_dates=aligned_dates,
        strategy_returns=strategy_returns,
        benchmark_returns=proxy_returns,
    )
    daily_curve_csv = _daily_curve_csv(
        folds=l3["folds"],
        aligned_dates=aligned_dates,
        strategy_returns=strategy_returns,
        benchmark_returns=proxy_returns,
    )
    fills_csv = _fills_csv(result.fills)
    target_met = bool(
        performance["annual_excess_return"] >= 0.15
        or (performance["annual_excess_return"] >= 0.10 and performance["excess_sharpe"] > 0.8)
    )
    stable: dict[str, Any] = {
        "status": "PASS_RESEARCH_ONLY",
        "target_met": target_met,
        "target_audit": {
            "basis": "FIVE_SEQUENTIAL_UNTOUCHED_OUTER_OOS_FOLDS",
            "criterion": (
                "annual_excess_return>=0.15 OR (annual_excess_return>=0.10 AND excess_sharpe>0.8)"
            ),
            "met": target_met,
            "if_not_met": (
                "DO_NOT_RETUNE_ON_OBSERVED_OUTER_OOS; redesign only on inner history and "
                "require new untouched forward data"
            ),
        },
        "initial_equity": str(initial_equity),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "sessions": len(sessions),
        "l3_content_hash": expected_l3_hash,
        "cache_content_hash": cache["content_hash"],
        "history_release_id": release_id,
        "code_version": args.code_version,
        "benchmark": proxy_metadata,
        "execution_assumptions": dict(_BASE_COST_ASSUMPTIONS),
        "outer_test_used_for_optimization": False,
        "l3_summary": {
            "fold_pool_hash": l3["fold_pool_hash"],
            "selected_method_counts": l3["selected_method_counts"],
            "outer_metrics_used_for_selection": False,
            "folds": [
                {
                    "fold": fold["fold"],
                    "family_count": fold["family_count"],
                    "inner_validation_start": fold["inner_validation_start"],
                    "inner_validation_end": fold["inner_validation_end"],
                    "outer_composite_rank_ic": fold["outer_composite_rank_ic"],
                }
                for fold in l3["folds"]
            ],
        },
        "fold_configurations": [
            {
                "fold": fold["fold"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "selected": fold["l4_selection"]["selected"],
            }
            for fold in l3["folds"]
        ],
        "fold_performance": fold_performance,
        "performance": performance,
        "artifacts": {
            "daily_outer_oos_curve": {
                "file": "nested_l4_daily_oos.csv",
                "sha256": _text_hash(daily_curve_csv),
            },
            "fills": {
                "file": "nested_l4_fills.csv",
                "sha256": _text_hash(fills_csv),
            },
        },
        "execution": {
            "orders": len(result.orders),
            "fills": len(result.fills),
            "fill_rate": str(metrics.fill_rate),
            "gross_turnover": str(metrics.gross_turnover),
            "total_fees": str(metrics.total_fees),
            "maximum_drawdown": str(metrics.max_drawdown),
            "final_equity": str(metrics.final_equity),
            "final_state_hash": result.final_state_hash,
            "t_plus_one_attested": t_plus_one,
        },
    }
    stable["content_hash"] = _hash(stable)
    payload = {**stable, "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "nested_l4_backtest.json", json.dumps(payload, indent=2) + "\n")
    _write(args.output_dir / "nested_l4_backtest.md", _markdown(payload))
    _write(args.output_dir / "nested_l4_daily_oos.csv", daily_curve_csv)
    _write(args.output_dir / "nested_l4_fills.csv", fills_csv)
    _write(
        args.output_dir / "benchmark_proxy.csv",
        "trade_date,return_rate,constituent_count\n"
        + "".join(
            f"{item.trade_date.isoformat()},{item.return_rate:.12g},{item.constituent_count}\n"
            for item in proxy
        ),
    )
    print(json.dumps({"status": payload["status"], "target_met": target_met, **performance}))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the five-year sequential outer-OOS L4")
    parser.add_argument("--history-release", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--l3-scores", type=Path, required=True)
    parser.add_argument("--l3-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    return parser.parse_args()


def _snapshots(
    *,
    trading_dates: tuple[date, ...],
    risk_snapshots: dict[date, Any],
    scores: np.ndarray[Any, Any],
    date_index: dict[date, int],
    code_index: dict[str, int],
) -> dict[date, SingleFactorSnapshot]:
    result: dict[date, SingleFactorSnapshot] = {}
    for trade_date in trading_dates:
        risk = risk_snapshots.get(trade_date)
        if risk is None:
            continue
        row = scores[date_index[trade_date]]
        observations = []
        for item in risk.observations:
            code = item.symbol.canonical.replace(".XSHG", ".SH").replace(".XSHE", ".SZ")
            position = code_index.get(code)
            value = float(row[position]) if position is not None else math.nan
            observations.append(
                SingleFactorObservation(
                    item.symbol,
                    trade_date,
                    risk.asof_time,
                    item.list_date,
                    value if math.isfinite(value) else None,
                    item.suspended,
                    item.is_st,
                    item.is_delisting_risk,
                )
            )
        result[trade_date] = SingleFactorSnapshot(trade_date, risk.asof_time, tuple(observations))
    return result


def _rebalance_configs(
    l3: dict[str, Any], session_dates: tuple[date, ...]
) -> dict[date, SingleFactorConfig]:
    result: dict[date, SingleFactorConfig] = {}
    for fold in l3["folds"]:
        start = date.fromisoformat(fold["test_start"])
        end = date.fromisoformat(fold["test_end"])
        fold_dates = tuple(day for day in session_dates if start <= day <= end)
        selected = fold["l4_selection"]["selected"]
        frequency = RebalanceFrequency(selected["frequency"])
        config = SingleFactorConfig(
            "nested_l3_size_neutral",
            "1.0.0",
            1,
            target_count=int(selected["target_count"]),
            minimum_constituents=max(30, int(selected["target_count"])),
            minimum_listing_days=120,
        )
        for trade_date in generate_rebalance_dates(fold_dates, frequency):
            result[trade_date] = config
    if not result:
        raise ValueError("nested L4 produced no rebalance configurations")
    return result


def _strategy_returns(equity_curve: tuple[Any, ...], initial: Decimal) -> dict[date, float]:
    previous = float(initial)
    result: dict[date, float] = {}
    for snapshot in equity_curve:
        equity = float(snapshot.equity)
        result[snapshot.asof_time.date()] = equity / previous - 1
        previous = equity
    return result


def _performance(
    strategy: np.ndarray[Any, Any], benchmark: np.ndarray[Any, Any]
) -> dict[str, float]:
    excess = strategy - benchmark
    annual_strategy = _annualized(strategy)
    annual_benchmark = _annualized(benchmark)
    strategy_standard_deviation = float(np.std(strategy, ddof=1))
    benchmark_standard_deviation = float(np.std(benchmark, ddof=1))
    excess_standard_deviation = float(np.std(excess, ddof=1))
    return {
        "annual_return": annual_strategy,
        "annual_benchmark_return": annual_benchmark,
        "annual_excess_return": annual_strategy - annual_benchmark,
        "annual_volatility": strategy_standard_deviation * np.sqrt(252),
        "annual_benchmark_volatility": benchmark_standard_deviation * np.sqrt(252),
        "tracking_error": excess_standard_deviation * np.sqrt(252),
        "excess_sharpe": (
            float(np.mean(excess) / excess_standard_deviation * np.sqrt(252))
            if excess_standard_deviation > 0
            else 0.0
        ),
        "maximum_drawdown": _maximum_drawdown(strategy),
        "benchmark_maximum_drawdown": _maximum_drawdown(benchmark),
        "relative_maximum_drawdown": _relative_maximum_drawdown(strategy, benchmark),
        "positive_excess_day_ratio": float(np.mean(excess > 0)),
        "total_return": float(np.prod(1 + strategy) - 1),
        "benchmark_total_return": float(np.prod(1 + benchmark) - 1),
    }


def _fold_performance(
    *,
    folds: list[dict[str, Any]],
    aligned_dates: tuple[date, ...],
    strategy_returns: dict[date, float],
    benchmark_returns: dict[date, float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fold in folds:
        start = date.fromisoformat(fold["test_start"])
        end = date.fromisoformat(fold["test_end"])
        dates = tuple(day for day in aligned_dates if start <= day <= end)
        if not dates:
            raise ValueError(f"outer fold {fold['fold']} has no aligned returns")
        performance = _performance(
            np.asarray([strategy_returns[day] for day in dates]),
            np.asarray([benchmark_returns[day] for day in dates]),
        )
        result.append(
            {
                "fold": fold["fold"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "sessions": len(dates),
                **performance,
            }
        )
    return result


def _daily_curve_csv(
    *,
    folds: list[dict[str, Any]],
    aligned_dates: tuple[date, ...],
    strategy_returns: dict[date, float],
    benchmark_returns: dict[date, float],
) -> str:
    ranges = tuple(
        (
            int(fold["fold"]),
            date.fromisoformat(fold["test_start"]),
            date.fromisoformat(fold["test_end"]),
        )
        for fold in folds
    )
    strategy_net_value = 1.0
    benchmark_net_value = 1.0
    rows = [
        "trade_date,outer_fold,strategy_return,benchmark_return,active_return,"
        "strategy_net_value,benchmark_net_value,relative_net_value\n"
    ]
    for trade_date in aligned_dates:
        fold = next(
            (number for number, start, end in ranges if start <= trade_date <= end),
            None,
        )
        if fold is None:
            raise ValueError(f"date outside sequential outer folds: {trade_date}")
        strategy_return = strategy_returns[trade_date]
        benchmark_return = benchmark_returns[trade_date]
        strategy_net_value *= 1 + strategy_return
        benchmark_net_value *= 1 + benchmark_return
        relative_net_value = strategy_net_value / benchmark_net_value
        rows.append(
            f"{trade_date.isoformat()},{fold},{strategy_return:.12g},"
            f"{benchmark_return:.12g},{strategy_return - benchmark_return:.12g},"
            f"{strategy_net_value:.12g},{benchmark_net_value:.12g},"
            f"{relative_net_value:.12g}\n"
        )
    return "".join(rows)


def _fills_csv(fills: tuple[Any, ...]) -> str:
    return "fill_id,order_id,occurred_at,symbol,side,quantity,price,fee,notional\n" + "".join(
        f"{fill.fill_id},{fill.order_id},{fill.occurred_at.isoformat()},"
        f"{fill.symbol.canonical},{fill.side.value},{fill.quantity},"
        f"{fill.price},{fill.fee},{fill.notional}\n"
        for fill in fills
    )


def _annualized(returns: np.ndarray[Any, Any]) -> float:
    total = float(np.prod(1 + returns))
    return total ** (252 / len(returns)) - 1 if total > 0 else -1.0


def _maximum_drawdown(returns: np.ndarray[Any, Any]) -> float:
    net_value = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], net_value)))
    return float(np.min(np.concatenate(([1.0], net_value)) / peaks - 1))


def _relative_maximum_drawdown(
    strategy: np.ndarray[Any, Any], benchmark: np.ndarray[Any, Any]
) -> float:
    benchmark_net_value = np.cumprod(1 + benchmark)
    if np.any(benchmark_net_value <= 0):
        raise ValueError("benchmark net value must remain positive")
    relative = np.cumprod(1 + strategy) / benchmark_net_value
    peaks = np.maximum.accumulate(np.concatenate(([1.0], relative)))
    return float(np.min(np.concatenate(([1.0], relative)) / peaks - 1))


def _markdown(payload: dict[str, Any]) -> str:
    performance = payload["performance"]
    fold_rows = "\n".join(
        "| {fold} | {test_start} | {test_end} | {sessions} | {annual_return:.2%} | "
        "{annual_benchmark_return:.2%} | {annual_excess_return:.2%} | {excess_sharpe:.3f} |".format(
            **fold
        )
        for fold in payload["fold_performance"]
    )
    configuration_rows = "\n".join(
        f"| {item['fold']} | {item['test_start']} | {item['test_end']} | "
        f"{item['selected']['target_count']} | {item['selected']['frequency']} |"
        for item in payload["fold_configurations"]
    )
    l3_rows = "\n".join(
        f"| {item['fold']} | {item['family_count']} | "
        f"{item['inner_validation_start']} | {item['inner_validation_end']} | "
        f"{_format_optional(item['outer_composite_rank_ic'])} |"
        for item in payload["l3_summary"]["folds"]
    )
    method_counts = json.dumps(payload["l3_summary"]["selected_method_counts"], sort_keys=True)
    return f"""# AQuant Nested Walk-Forward L4 Backtest

- Status: `{payload["status"]}`
- Period: {payload["start_date"]} to {payload["end_date"]}
- Initial equity: CNY {payload["initial_equity"]}
- Code version: `{payload["code_version"]}`
- Annual return: {performance["annual_return"]:.2%}
- Proxy annual return: {performance["annual_benchmark_return"]:.2%}
- Annual excess return: {performance["annual_excess_return"]:.2%}
- Excess Sharpe: {performance["excess_sharpe"]:.3f}
- Tracking error: {performance["tracking_error"]:.2%}
- Maximum drawdown: {performance["maximum_drawdown"]:.2%}
- Relative maximum drawdown: {performance["relative_maximum_drawdown"]:.2%}
- Target met: `{payload["target_met"]}`
- Target basis: `{payload["target_audit"]["basis"]}`
- Outer test used for optimization: `False`
- T+1 attested: `{payload["execution"]["t_plus_one_attested"]}`

## Execution Assumptions

- Signal / fill: T close / T+1 open
- Slippage: 5 bps per fill
- Commission: 3 bps, minimum CNY 5 per fill
- Sell stamp duty: 10 bps before 2023-08-28; 5 bps thereafter
- Transfer fee: 0.2 bps before 2022-04-29; 0.1 bps thereafter
- Lot / liquidity cap: 100 shares; 0.5% of prior 20-day average volume
- Suspensions and price-limit blocks are enforced; unfilled orders cancel after the open

## Sequential Outer-OOS Folds

| Fold | Start | End | Sessions | Strategy p.a. | Proxy p.a. | Excess p.a. | Excess Sharpe |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
{fold_rows}

## Frozen L4 Configuration by Fold

| Fold | Start | End | Holdings | Rebalance |
| --- | --- | --- | ---: | --- |
{configuration_rows}

## L3 Audit

- Selected family-method counts: `{method_counts}`
- Outer metrics used for selection: `False`

| Fold | Families | Inner validation start | Inner validation end | Outer RankIC (report only) |
| --- | ---: | --- | --- | ---: |
{l3_rows}

## Methodological Limits

- Benchmark is `PIT_ALL_A_SHARE_DAILY_EQUAL_PROXY`, not the official Wind All-A index.
- Inner L4 selection uses an available-security equal-weight close-return proxy; the final
  event backtest uses the PIT all-A proxy documented in the JSON metadata.
- If the locked outer-OOS target is missed, those observations must not be used to tune
  this version. A redesigned version requires new untouched forward evidence.
"""


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _format_optional(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
