#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from aquant.backtest import (
    EventDrivenBacktest,
    UnfilledOrderPolicy,
    calculate_metrics,
)
from aquant.data.history import DuckDBMicrocapHistory
from aquant.factors.atomic import baseline_factor_library, second_wave_candidate_library
from aquant.factors.materialization import MaterializedFactorReader
from aquant.factors.spec import FactorSpec
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


def _date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYYMMDD") from exc


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
        raise ValueError("single-factor L4 backtest requires one admitted Production factor")
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


def _markdown(payload: dict[str, object]) -> str:
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    return "\n".join(
        (
            "# Single-factor L4 backtest",
            "",
            f"- Status: {payload['status']}",
            f"- Run ID: `{payload['run_id']}`",
            f"- Factor: `{payload['factor_id']}@{payload['factor_version']}`",
            f"- Period: {payload['start_date']} to {payload['end_date']}",
            f"- Sessions: {payload['sessions']}",
            f"- Rebalances: {payload['rebalance_count']}",
            f"- Orders / fills: {payload['orders']} / {payload['fills']}",
            f"- Total return: {Decimal(str(metrics['total_return'])):.4%}",
            f"- Maximum drawdown: {Decimal(str(metrics['maximum_drawdown'])):.4%}",
            f"- Gross turnover: {Decimal(str(metrics['gross_turnover'])):.4f}",
            f"- Total fees: {metrics['total_fees']}",
            f"- Fill rate: {Decimal(str(metrics['fill_rate'])):.2%}",
            f"- Final state hash: `{payload['final_state_hash']}`",
            f"- Content hash: `{payload['content_hash']}`",
            "",
            "This is a single-factor research chain validation, not an approved "
            "multi-factor production strategy.",
            "",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a PIT-safe, T+1 single-factor L4 research backtest"
    )
    parser.add_argument("--history-release", type=Path, required=True)
    parser.add_argument("--factor-materialization", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--factor-id", default="amount_concentration_20d")
    parser.add_argument("--factor-version", default="1.0.0")
    parser.add_argument("--start-date", type=_date, required=True)
    parser.add_argument("--end-date", type=_date, required=True)
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--minimum-constituents", type=int, default=30)
    parser.add_argument("--minimum-listing-days", type=int, default=120)
    parser.add_argument("--initial-equity", type=Decimal, default=Decimal("10000000"))
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
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/l4_single_factor"))
    args = parser.parse_args(argv)
    try:
        if args.start_date >= args.end_date:
            raise ValueError("backtest requires at least two ordered dates")
        spec = _factor_spec(args.factor_id, args.factor_version)
        expected_direction = int(spec.expected_direction)
        admission = _verified_admission(
            args.admission,
            args.factor_id,
            args.factor_version,
        )
        factor_reader = MaterializedFactorReader(args.factor_materialization)
        if factor_reader.manifest.data_release_id != admission["data_release_id"]:
            raise ValueError("factor materialization and admission data releases differ")
        stored = factor_reader.load(
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
            risk_snapshots = history.snapshots(dates)
            history_release_id = history.release.manifest.release_id
            history_release_hash = _hash(
                json.loads((args.history_release / "manifest.json").read_text(encoding="utf-8"))
            )
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

        config = SingleFactorConfig(
            factor_id=args.factor_id,
            factor_version=args.factor_version,
            expected_direction=expected_direction,
            target_count=args.target_count,
            minimum_constituents=args.minimum_constituents,
            minimum_listing_days=args.minimum_listing_days,
        )
        frequency = RebalanceFrequency(args.frequency)
        rebalance_dates = generate_rebalance_dates(dates, frequency)
        if not rebalance_dates:
            raise ValueError("backtest range produced no eligible rebalance dates")
        first_selection = SingleFactorSelector(config).select(snapshots[min(rebalance_dates)])
        stable_config = {
            "admission_hash": admission["content_hash"],
            "cost_scenario": args.cost_scenario,
            "data_release_id": factor_reader.manifest.data_release_id,
            "end_date": args.end_date.isoformat(),
            "expected_direction": expected_direction,
            "factor_id": args.factor_id,
            "factor_materialization_hash": factor_reader.manifest.content_hash,
            "factor_version": args.factor_version,
            "frequency": frequency.value,
            "history_release_hash": history_release_hash,
            "initial_equity": str(args.initial_equity),
            "minimum_constituents": args.minimum_constituents,
            "minimum_listing_days": args.minimum_listing_days,
            "start_date": args.start_date.isoformat(),
            "target_count": args.target_count,
        }
        config_hash = _hash(stable_config)
        run_id = f"single-factor-l4-{args.factor_id}-{config_hash[:12]}"
        result = EventDrivenBacktest(
            run_id=run_id,
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
        order_by_id = {order.order_id: order for order in result.orders}
        t_plus_one = all(
            fill.occurred_at.date() > order_by_id[fill.order_id].submitted_at.date()
            for fill in result.fills
        )
        if not result.fills or not t_plus_one:
            raise ValueError("single-factor L4 chain produced no fills or violated T+1")
        stable: dict[str, object] = {
            **stable_config,
            "config_hash": config_hash,
            "corporate_actions": len(result.corporate_actions),
            "factor_rows": len(stored),
            "fills": len(result.fills),
            "final_state_hash": result.final_state_hash,
            "first_selection": {
                "eligible_count": first_selection.eligible_count,
                "excluded_counts": dict(first_selection.excluded_counts),
                "selected_count": len(first_selection.symbols),
            },
            "history_release_id": history_release_id,
            "metrics": {
                "fill_rate": str(metrics.fill_rate),
                "final_equity": str(metrics.final_equity),
                "gross_turnover": str(metrics.gross_turnover),
                "maximum_drawdown": str(metrics.max_drawdown),
                "total_fees": str(metrics.total_fees),
                "total_return": str(metrics.total_return),
            },
            "orders": len(result.orders),
            "rebalance_count": len(rebalance_dates),
            "run_id": run_id,
            "sessions": len(sessions),
            "status": "PASS_RESEARCH_ONLY",
            "t_plus_one_attested": t_plus_one,
        }
        stable["content_hash"] = _hash(stable)
        payload = {
            **stable,
            "created_at": datetime.now(UTC).isoformat(timespec="microseconds"),
            "warning": (
                "Single-factor L4 chain validation only; not an approved "
                "multi-factor production strategy."
            ),
        }
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
                    "status": payload["status"],
                    "run_id": run_id,
                    "report": str(json_path),
                    "content_hash": payload["content_hash"],
                    "sessions": len(sessions),
                    "orders": len(result.orders),
                    "fills": len(result.fills),
                    "total_return": str(metrics.total_return),
                    "maximum_drawdown": str(metrics.max_drawdown),
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
