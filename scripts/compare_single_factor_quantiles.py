#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import tinyshare as ts  # type: ignore[import-untyped]
from dotenv import load_dotenv

from aquant.backtest import EventDrivenBacktest, UnfilledOrderPolicy, calculate_metrics
from aquant.data.history import DuckDBMicrocapHistory
from aquant.factors.atomic import baseline_factor_library, second_wave_candidate_library
from aquant.factors.materialization import MaterializedFactorReader
from aquant.factors.spec import FactorSpec
from aquant.strategies import (
    CostScenario,
    RebalanceFrequency,
    SelectionTail,
    SingleFactorConfig,
    SingleFactorEqualWeightStrategy,
    SingleFactorObservation,
    SingleFactorSelector,
    SingleFactorSnapshot,
    generate_rebalance_dates,
    matcher_for_cost_scenario,
    monthly_benchmark_results,
    monthly_portfolio_results,
)


def _date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYYMMDD") from exc


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


def _verified_admission(path: Path, factor_id: str, factor_version: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("content_hash", None)
    if expected != _hash(payload):
        raise ValueError("admission artifact content hash mismatch")
    matches = [
        item
        for item in payload["factors"]
        if item["factor_id"] == factor_id and item["factor_version"] == factor_version
    ]
    if len(matches) != 1 or matches[0]["lifecycle_status"] != "PRODUCTION":
        raise ValueError("quantile comparison requires one admitted Production factor")
    return {**payload, "content_hash": expected}


def _factor_spec(factor_id: str, factor_version: str) -> FactorSpec:
    matches = [
        factor.spec
        for factor in (*baseline_factor_library(), *second_wave_candidate_library())
        if factor.spec.factor_id == factor_id and factor.spec.version == factor_version
    ]
    if len(matches) != 1:
        raise ValueError("single-factor definition is missing or ambiguous")
    return matches[0]


def _write_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _benchmark_closes(
    *,
    benchmark_code: str,
    start_date: date,
    end_date: date,
) -> dict[date, Decimal]:
    load_dotenv()
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise ValueError("TUSHARE_TOKEN is required to retrieve the benchmark")
    query_start = start_date - timedelta(days=40)
    frame = ts.pro_api(token).index_daily(
        ts_code=benchmark_code,
        start_date=query_start.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
    )
    required = {"ts_code", "trade_date", "close"}
    if not required.issubset(frame.columns):
        raise ValueError("benchmark response is missing required columns")
    closes = {
        datetime.strptime(str(row.trade_date), "%Y%m%d").date(): Decimal(str(row.close))
        for row in frame.itertuples(index=False)
        if str(row.ts_code) == benchmark_code and Decimal(str(row.close)) > 0
    }
    if max(closes, default=date.min) < end_date:
        raise ValueError("benchmark does not cover the requested common end date")
    return closes


def _markdown(payload: dict[str, object]) -> str:
    rows = payload["monthly"]
    assert isinstance(rows, list)
    lines = [
        "# Single-factor 20% quantile comparison",
        "",
        f"- Factor: `{payload['factor_id']}@{payload['factor_version']}`",
        f"- Period: {payload['start_date']} to {payload['end_date']}",
        f"- Benchmark: `{payload['benchmark_code']}`",
        f"- Frequency: `{payload['frequency']}`",
        f"- Initial equity per portfolio: {payload['initial_equity']}",
        f"- Benchmark total return: {Decimal(str(payload['benchmark_total_return'])):.4%}",
        f"- Best minus worst: {Decimal(str(payload['best_minus_worst_total_return'])):.4%}",
        f"- Best excess benchmark: "
        f"{Decimal(str(payload['best_excess_benchmark_total_return'])):.4%}",
        "",
        "| Month | Best 20% net | Worst 20% net | Benchmark | Best turnover "
        "| Worst turnover | Best friction | Worst friction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        assert isinstance(row, dict)
        lines.append(
            f"| {row['month']} | {Decimal(str(row['best_net_return'])):.4%} "
            f"| {Decimal(str(row['worst_net_return'])):.4%} "
            f"| {Decimal(str(row['benchmark_return'])):.4%} "
            f"| {Decimal(str(row['best_turnover'])):.4f} "
            f"| {Decimal(str(row['worst_turnover'])):.4f} "
            f"| {Decimal(str(row['best_friction_cost'])):.2f} "
            f"| {Decimal(str(row['worst_friction_cost'])):.2f} |"
        )
    lines.extend(
        (
            "",
            "Best means the factor's admitted expected direction; for this negative-direction "
            "factor it is the lowest raw-value 20%. Worst is the highest raw-value 20%.",
            "",
            "Returns are after actual fees and slippage. Turnover is monthly filled notional "
            "divided by month-opening equity. Friction includes fees plus modeled slippage.",
            "",
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare best/worst factor quintiles with CSI All")
    parser.add_argument("--history-release", type=Path, required=True)
    parser.add_argument("--factor-materialization", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--factor-id", default="amount_concentration_20d")
    parser.add_argument("--factor-version", default="1.0.0")
    parser.add_argument("--start-date", type=_date, required=True)
    parser.add_argument("--end-date", type=_date, required=True)
    parser.add_argument("--target-fraction", type=Decimal, default=Decimal("0.2"))
    parser.add_argument("--minimum-constituents", type=int, default=30)
    parser.add_argument("--minimum-listing-days", type=int, default=120)
    parser.add_argument("--initial-equity", type=Decimal, default=Decimal("1000000000"))
    parser.add_argument("--benchmark-code", default="000985.CSI")
    parser.add_argument(
        "--frequency",
        choices=[item.value for item in RebalanceFrequency],
        default=RebalanceFrequency.WEEKLY.value,
    )
    parser.add_argument(
        "--cost-scenario",
        choices=[item.value for item in CostScenario],
        default=CostScenario.BASE_COST.value,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/l4_single_factor_quantiles"),
    )
    args = parser.parse_args(argv)
    try:
        if args.start_date >= args.end_date:
            raise ValueError("comparison requires at least two ordered dates")
        spec = _factor_spec(args.factor_id, args.factor_version)
        admission = _verified_admission(args.admission, args.factor_id, args.factor_version)
        reader = MaterializedFactorReader(args.factor_materialization)
        if reader.manifest.data_release_id != admission["data_release_id"]:
            raise ValueError("factor materialization and admission data releases differ")
        stored = reader.load(
            factor_id=args.factor_id,
            factor_version=args.factor_version,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        values_by_date: dict[date, dict[str, float]] = {}
        for item in stored:
            values_by_date.setdefault(item.trade_date, {})[item.ts_code] = item.value

        with DuckDBMicrocapHistory(args.history_release) as history:
            sessions = history.sessions(args.start_date, args.end_date)
            dates = tuple(session.trade_date for session in sessions)
            if not dates or dates[-1] != args.end_date:
                raise ValueError("history does not end on the requested common end date")
            risk_snapshots = history.snapshots(dates)
            history_release_id = history.release.manifest.release_id
        snapshots: dict[date, SingleFactorSnapshot] = {}
        for trade_date in dates:
            risk = risk_snapshots[trade_date]
            daily_values = values_by_date.get(trade_date, {})
            snapshots[trade_date] = SingleFactorSnapshot(
                trade_date,
                risk.asof_time,
                tuple(
                    SingleFactorObservation(
                        symbol=item.symbol,
                        trade_date=trade_date,
                        available_at=risk.asof_time,
                        list_date=item.list_date,
                        factor_value=daily_values.get(
                            item.symbol.canonical.replace(".XSHG", ".SH").replace(".XSHE", ".SZ")
                        ),
                        suspended=item.suspended,
                        is_st=item.is_st,
                        is_delisting_risk=item.is_delisting_risk,
                    )
                    for item in risk.observations
                ),
            )

        frequency = RebalanceFrequency(args.frequency)
        rebalance_dates = generate_rebalance_dates(dates, frequency)
        if not rebalance_dates:
            raise ValueError("comparison range produced no eligible rebalance dates")
        monthly_by_tail: dict[SelectionTail, dict[str, dict[str, str]]] = {}
        summary_by_tail: dict[str, dict[str, object]] = {}
        for tail in SelectionTail:
            config = SingleFactorConfig(
                factor_id=args.factor_id,
                factor_version=args.factor_version,
                expected_direction=int(spec.expected_direction),
                target_fraction=args.target_fraction,
                selection_tail=tail,
                minimum_constituents=args.minimum_constituents,
                minimum_listing_days=args.minimum_listing_days,
            )
            first_selection = SingleFactorSelector(config).select(snapshots[min(rebalance_dates)])
            result = EventDrivenBacktest(
                run_id=f"single-factor-{tail.value}-quintile",
                initial_cash=args.initial_equity,
                matcher=matcher_for_cost_scenario(CostScenario(args.cost_scenario)),
                unfilled_order_policy=UnfilledOrderPolicy.CANCEL_AFTER_OPEN,
            ).run(
                sessions,
                SingleFactorEqualWeightStrategy(
                    snapshots=snapshots,
                    rebalance_dates=rebalance_dates,
                    config=config,
                ),
            )
            metrics = calculate_metrics(result, initial_equity=args.initial_equity)
            monthly_by_tail[tail] = {
                item.month: item.as_dict()
                for item in monthly_portfolio_results(
                    result,
                    initial_equity=args.initial_equity,
                    sessions=sessions,
                )
            }
            summary_by_tail[tail.value] = {
                "eligible_count": first_selection.eligible_count,
                "selected_count": len(first_selection.symbols),
                "total_return": str(metrics.total_return),
                "gross_turnover": str(metrics.gross_turnover),
                "total_fees": str(metrics.total_fees),
                "maximum_drawdown": str(metrics.max_drawdown),
                "orders": len(result.orders),
                "fills": len(result.fills),
                "final_state_hash": result.final_state_hash,
            }

        benchmark_closes = _benchmark_closes(
            benchmark_code=args.benchmark_code,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        benchmark = {
            item.month: item
            for item in monthly_benchmark_results(
                benchmark_closes,
                start_date=args.start_date,
                end_date=args.end_date,
            )
        }
        common_months = sorted(
            set(monthly_by_tail[SelectionTail.BEST])
            & set(monthly_by_tail[SelectionTail.WORST])
            & set(benchmark)
        )
        monthly = []
        for month in common_months:
            best = monthly_by_tail[SelectionTail.BEST][month]
            worst = monthly_by_tail[SelectionTail.WORST][month]
            monthly.append(
                {
                    "month": month,
                    "best_net_return": best["net_return"],
                    "worst_net_return": worst["net_return"],
                    "benchmark_return": str(benchmark[month].return_rate),
                    "best_turnover": best["turnover"],
                    "worst_turnover": worst["turnover"],
                    "best_fees": best["fees"],
                    "worst_fees": worst["fees"],
                    "best_slippage": best["slippage"],
                    "worst_slippage": worst["slippage"],
                    "best_friction_cost": best["friction_cost"],
                    "worst_friction_cost": worst["friction_cost"],
                    "best_friction_bps": best["friction_bps"],
                    "worst_friction_bps": worst["friction_bps"],
                    "best_minus_worst": str(
                        Decimal(best["net_return"]) - Decimal(worst["net_return"])
                    ),
                    "best_excess_benchmark": str(
                        Decimal(best["net_return"]) - benchmark[month].return_rate
                    ),
                }
            )
        benchmark_growth = Decimal("1")
        for month in common_months:
            benchmark_growth *= Decimal("1") + benchmark[month].return_rate
        benchmark_total_return = benchmark_growth - Decimal("1")
        best_total_return = Decimal(str(summary_by_tail[SelectionTail.BEST.value]["total_return"]))
        worst_total_return = Decimal(
            str(summary_by_tail[SelectionTail.WORST.value]["total_return"])
        )
        stable: dict[str, object] = {
            "status": "PASS_RESEARCH_ONLY",
            "factor_id": args.factor_id,
            "factor_version": args.factor_version,
            "expected_direction": int(spec.expected_direction),
            "start_date": args.start_date.isoformat(),
            "end_date": args.end_date.isoformat(),
            "frequency": frequency.value,
            "cost_scenario": args.cost_scenario,
            "target_fraction": str(args.target_fraction),
            "initial_equity": str(args.initial_equity),
            "benchmark_code": args.benchmark_code,
            "benchmark_daily_hash": _hash(
                {day.isoformat(): str(value) for day, value in sorted(benchmark_closes.items())}
            ),
            "data_release_id": reader.manifest.data_release_id,
            "history_release_id": history_release_id,
            "factor_materialization_hash": reader.manifest.content_hash,
            "admission_hash": admission["content_hash"],
            "sessions": len(sessions),
            "factor_rows": len(stored),
            "rebalance_count": len(rebalance_dates),
            "portfolios": summary_by_tail,
            "benchmark_total_return": str(benchmark_total_return),
            "best_minus_worst_total_return": str(best_total_return - worst_total_return),
            "best_excess_benchmark_total_return": str(best_total_return - benchmark_total_return),
            "monthly": monthly,
        }
        content_hash = _hash(stable)
        stable["content_hash"] = content_hash
        payload = {**stable, "created_at": datetime.now(UTC).isoformat(timespec="microseconds")}
        run_id = f"single-factor-quintiles-{args.factor_id}-{content_hash[:12]}"
        json_path = args.output_dir / f"{run_id}.json"
        markdown_path = args.output_dir / f"{run_id}.md"
        _write_atomic(
            json_path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        _write_atomic(markdown_path, _markdown(payload))
        print(
            json.dumps(
                {
                    "status": stable["status"],
                    "report": str(json_path),
                    "content_hash": stable["content_hash"],
                    "portfolios": summary_by_tail,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
