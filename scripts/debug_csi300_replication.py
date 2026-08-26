#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import yaml

from aquant.backtest import (
    CorporateActionKind,
    EventDrivenBacktest,
    MarketSession,
    NextOpenMatcher,
    calculate_metrics,
)
from aquant.data.history import DuckDBMicrocapHistory
from aquant.data.history.tushare import TushareHistoryCatalog
from aquant.data.index_history import IndexHistory, IndexWeightSnapshot, load_index_history
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar
from aquant.strategies import IndexRebalanceTarget, IndexReplicationStrategy


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


def _write_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _next_session(day: date, trading_dates: Sequence[date]) -> date:
    try:
        return next(value for value in trading_dates if value > day)
    except StopIteration as exc:
        raise ValueError(f"no trading session follows {day}") from exc


def _session_on_or_after(day: date, trading_dates: Sequence[date]) -> date:
    try:
        return next(value for value in trading_dates if value >= day)
    except StopIteration as exc:
        raise ValueError(f"no trading session exists on or after {day}") from exc


def _previous_session(day: date, trading_dates: Sequence[date]) -> date:
    prior = [value for value in trading_dates if value < day]
    if not prior:
        raise ValueError(f"no trading session precedes {day}")
    return prior[-1]


def _second_friday(year: int, month: int) -> date:
    candidates = [
        date(year, month, day) for day in range(8, 15) if date(year, month, day).weekday() == 4
    ]
    if len(candidates) != 1:
        raise ValueError("unable to resolve second Friday")
    return candidates[0]


def _schedule(
    index: IndexHistory,
    trading_dates: tuple[date, ...],
    *,
    start_date: date,
    end_date: date,
    temporary_effective_dates: Mapping[str, date],
) -> tuple[tuple[date, IndexWeightSnapshot, str], ...]:
    snapshots = index.weight_snapshots
    initial_effective = _next_session(date(start_date.year - 1, 12, 28), trading_dates)
    initial = max(
        (item for item in snapshots if item.trade_date <= initial_effective),
        key=lambda item: item.trade_date,
    )
    events: list[tuple[date, IndexWeightSnapshot, str]] = [(initial_effective, initial, "INITIAL")]
    regular_sources: set[date] = set()
    for year in range(start_date.year, end_date.year + 1):
        for month in (6, 12):
            effective = _next_session(_second_friday(year, month), trading_dates)
            if effective > end_date:
                continue
            source = next(item for item in snapshots if item.trade_date >= effective)
            events.append((effective, source, "REGULAR_SEMIANNUAL"))
            regular_sources.add(source.trade_date)

    previous = snapshots[0]
    for current in snapshots[1:]:
        prior_symbols = {symbol for symbol, _ in previous.weights}
        current_symbols = {symbol for symbol, _ in current.weights}
        changed = prior_symbols != current_symbols
        if (
            changed
            and start_date <= current.trade_date <= end_date
            and current.trade_date not in regular_sources
            and current.trade_date != initial.trade_date
        ):
            outgoing = prior_symbols - current_symbols
            audited_dates = {
                temporary_effective_dates[
                    symbol.canonical.replace(".XSHG", ".SH").replace(".XSHE", ".SZ")
                ]
                for symbol in outgoing
                if symbol.canonical.replace(".XSHG", ".SH").replace(".XSHE", ".SZ")
                in temporary_effective_dates
            }
            effective = (
                _session_on_or_after(audited_dates.pop(), trading_dates)
                if len(audited_dates) == 1
                else current.trade_date
            )
            events.append(
                (
                    effective,
                    current,
                    (
                        "AUDITED_TEMPORARY_CHANGE"
                        if effective != current.trade_date
                        else "OBSERVED_TEMPORARY_CHANGE"
                    ),
                )
            )
        previous = current
    return tuple(sorted(events, key=lambda item: item[0]))


def _last_prices(
    sessions: Sequence[MarketSession],
) -> dict[date, dict[Symbol, Decimal]]:
    output: dict[date, dict[Symbol, Decimal]] = {}
    visible: dict[Symbol, Decimal] = {}
    for session in sessions:
        visible.update((bar.symbol, bar.close) for bar in session.bars)
        output[session.trade_date] = dict(visible)
    return output


def _targets(
    schedule: Sequence[tuple[date, IndexWeightSnapshot, str]],
    sessions: Sequence[MarketSession],
) -> dict[date, IndexRebalanceTarget]:
    trading_dates = tuple(session.trade_date for session in sessions)
    by_date = {session.trade_date: session for session in sessions}
    visible_closes = _last_prices(sessions)
    targets: dict[date, IndexRebalanceTarget] = {}
    for effective, snapshot, reason in schedule:
        signal = _previous_session(effective, trading_dates)
        bars = by_date[effective].bar_by_symbol()
        share_multipliers: dict[Symbol, Decimal] = {}
        for session in sessions:
            if not effective < session.trade_date <= snapshot.trade_date:
                continue
            for action in session.corporate_actions:
                if action.kind in {
                    CorporateActionKind.STOCK_DIVIDEND,
                    CorporateActionKind.CAPITALIZATION,
                }:
                    share_multipliers[action.symbol] = share_multipliers.get(
                        action.symbol, Decimal("1")
                    ) * (Decimal("1") + action.ratio)
                elif action.kind in {
                    CorporateActionKind.SPLIT,
                    CorporateActionKind.REVERSE_SPLIT,
                }:
                    share_multipliers[action.symbol] = (
                        share_multipliers.get(action.symbol, Decimal("1")) * action.ratio
                    )
        scores: list[tuple[Symbol, Decimal]] = []
        for symbol, source_weight in snapshot.weights:
            source_price = visible_closes[snapshot.trade_date][symbol]
            effective_price = (
                bars[symbol].open if symbol in bars else visible_closes[signal][symbol]
            )
            effective_share_unit = (
                source_weight / source_price / share_multipliers.get(symbol, Decimal("1"))
            )
            scores.append((symbol, effective_share_unit * effective_price))
        normalized_total = sum((score for _, score in scores), Decimal("0"))
        weights = tuple((symbol, score / normalized_total) for symbol, score in scores)
        prices = tuple(
            (
                symbol,
                bars[symbol].open if symbol in bars else visible_closes[signal][symbol],
            )
            for symbol, _ in weights
        )
        valuation_prices = tuple(
            (
                symbol,
                bars[symbol].open if symbol in bars else visible_closes[signal][symbol],
            )
            for symbol in sorted(visible_closes[signal])
        )
        targets[signal] = IndexRebalanceTarget(
            signal_date=signal,
            effective_date=effective,
            weights=weights,
            execution_prices=prices,
            source_snapshot_date=snapshot.trade_date,
            reason=reason,
            portfolio_valuation_prices=valuation_prices,
        )
    return targets


def _price_index_sessions(sessions: Sequence[MarketSession]) -> tuple[MarketSession, ...]:
    ignored = {
        CorporateActionKind.CASH_DIVIDEND_ENTITLEMENT,
        CorporateActionKind.CASH_DIVIDEND_PAYMENT,
    }
    output: list[MarketSession] = []
    visible: dict[Symbol, Decimal] = {}
    for session in sessions:
        existing = session.bar_by_symbol()
        visible.update((symbol, bar.close) for symbol, bar in existing.items())
        synthetic = tuple(
            DailyBar(
                symbol,
                session.trade_date,
                price,
                price,
                price,
                price,
                Decimal("0"),
                Decimal("0"),
            )
            for symbol, price in visible.items()
            if symbol not in existing
        )
        output.append(
            MarketSession(
                session.trade_date,
                session.open_at,
                session.close_at,
                (*session.bars, *synthetic),
                session.statuses,
                tuple(action for action in session.corporate_actions if action.kind not in ignored),
            )
        )
    return tuple(output)


def _period_returns(
    values: Mapping[date, Decimal],
    *,
    start_date: date,
    end_date: date,
    annual: bool,
) -> dict[str, Decimal]:
    ordered = sorted(values.items())
    prior = [value for day, value in ordered if day < start_date]
    if not prior:
        raise ValueError("return series requires one observation before start date")
    previous = prior[-1]
    ends: dict[str, Decimal] = {}
    for day, value in ordered:
        if start_date <= day <= end_date:
            key = str(day.year) if annual else day.strftime("%Y-%m")
            ends[key] = value
    output: dict[str, Decimal] = {}
    for key, ending in sorted(ends.items()):
        output[key] = ending / previous - Decimal("1")
        previous = ending
    return output


def _tracking_error(
    portfolio: Mapping[date, Decimal],
    benchmark: Mapping[date, Decimal],
    *,
    start_date: date,
    end_date: date,
) -> float:
    common = sorted(set(portfolio) & set(benchmark))
    common = [day for day in common if start_date <= day <= end_date]
    differences: list[float] = []
    for previous, current in pairwise(common):
        portfolio_return = portfolio[current] / portfolio[previous] - Decimal("1")
        benchmark_return = benchmark[current] / benchmark[previous] - Decimal("1")
        differences.append(float(portfolio_return - benchmark_return))
    if len(differences) < 2:
        raise ValueError("tracking error requires at least two aligned returns")
    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
    return math.sqrt(variance * 252)


def _markdown(payload: dict[str, object]) -> str:
    monthly = payload["monthly"]
    annual = payload["annual"]
    assert isinstance(monthly, list)
    assert isinstance(annual, list)
    lines = [
        "# CSI 300 backtest-engine replication debug",
        "",
        f"- Period: {payload['start_date']} to {payload['end_date']}",
        f"- Index: `{payload['index_code']}`",
        f"- Rebalances: {payload['rebalance_count']}",
        f"- Annualized daily tracking error: {float(Decimal(str(payload['tracking_error']))):.4%}",
        "",
        "| Month | Replication | CSI 300 | Difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in monthly:
        assert isinstance(row, dict)
        lines.append(
            f"| {row['period']} | {Decimal(str(row['replication'])):.4%} "
            f"| {Decimal(str(row['benchmark'])):.4%} "
            f"| {Decimal(str(row['difference'])):.4%} |"
        )
    lines.extend(
        ("", "| Year | Replication | CSI 300 | Difference |", "| --- | ---: | ---: | ---: |")
    )
    for row in annual:
        assert isinstance(row, dict)
        lines.append(
            f"| {row['period']} | {Decimal(str(row['replication'])):.4%} "
            f"| {Decimal(str(row['benchmark'])):.4%} "
            f"| {Decimal(str(row['difference'])):.4%} |"
        )
    lines.extend(
        (
            "",
            "Calibration conventions: zero fees/slippage, next-open fills, no limit/status "
            "blocking, cash dividends removed to match the price index, and stock actions "
            "retained.",
            "",
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replicate CSI 300 to debug the backtest engine")
    parser.add_argument("--history-release", type=Path, required=True)
    parser.add_argument("--start-date", type=_date, default=date(2024, 1, 1))
    parser.add_argument("--end-date", type=_date, default=date(2025, 12, 31))
    parser.add_argument("--index-code", default="000300.SH")
    parser.add_argument(
        "--schedule-config",
        type=Path,
        default=Path("config/index_replication/csi300_2024_2025.yaml"),
    )
    parser.add_argument("--initial-equity", type=Decimal, default=Decimal("10000000000"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/backtest_debug/csi300"),
    )
    args = parser.parse_args(argv)
    try:
        manifest_path = args.history_release / "manifest.json"
        release_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schedule_config = yaml.safe_load(args.schedule_config.read_text(encoding="utf-8"))
        if schedule_config.get("index_code") != args.index_code:
            raise ValueError("index schedule configuration does not match index code")
        temporary_effective_dates = {
            str(item["outgoing"]): _date(str(item["effective_date"]))
            for item in schedule_config.get("temporary_adjustments", [])
        }
        catalog = TushareHistoryCatalog(Path(release_manifest["raw_state_path"]))
        index = load_index_history(
            catalog,
            index_code=args.index_code,
            start_date=date(args.start_date.year - 1, 12, 1),
            end_date=args.end_date,
        )
        all_symbols = tuple(
            sorted(
                {symbol for snapshot in index.weight_snapshots for symbol, _ in snapshot.weights}
            )
        )
        ts_codes = tuple(
            f"{symbol.code}.{'SH' if symbol.exchange.value == 'XSHG' else 'SZ'}"
            for symbol in all_symbols
        )
        session_start = date(args.start_date.year - 1, 12, 1)
        with DuckDBMicrocapHistory(args.history_release) as history:
            sessions = history.sessions(
                session_start,
                args.end_date,
                ts_codes=ts_codes,
            )
            release_id = history.release.manifest.release_id
        trading_dates = tuple(session.trade_date for session in sessions)
        schedule = _schedule(
            index,
            trading_dates,
            start_date=args.start_date,
            end_date=args.end_date,
            temporary_effective_dates=temporary_effective_dates,
        )
        targets = _targets(schedule, sessions)
        result = EventDrivenBacktest(
            run_id="csi300-price-index-replication",
            initial_cash=args.initial_equity,
            matcher=NextOpenMatcher(),
        ).run(
            _price_index_sessions(sessions),
            IndexReplicationStrategy(targets, cash_buffer_weight=Decimal("0")),
        )
        unfilled = [order for order in result.orders if order.remaining_quantity]
        if unfilled:
            raise ValueError(
                "index calibration left unfilled orders: "
                f"{len(unfilled)}; first={unfilled[0].symbol.canonical}"
            )
        metrics = calculate_metrics(result, initial_equity=args.initial_equity)
        portfolio = {snapshot.asof_time.date(): snapshot.equity for snapshot in result.equity_curve}
        benchmark = dict(index.closes)
        portfolio_monthly = _period_returns(
            portfolio,
            start_date=args.start_date,
            end_date=args.end_date,
            annual=False,
        )
        benchmark_monthly = _period_returns(
            benchmark,
            start_date=args.start_date,
            end_date=args.end_date,
            annual=False,
        )
        portfolio_annual = _period_returns(
            portfolio,
            start_date=args.start_date,
            end_date=args.end_date,
            annual=True,
        )
        benchmark_annual = _period_returns(
            benchmark,
            start_date=args.start_date,
            end_date=args.end_date,
            annual=True,
        )
        monthly = [
            {
                "period": period,
                "replication": str(portfolio_monthly[period]),
                "benchmark": str(benchmark_monthly[period]),
                "difference": str(portfolio_monthly[period] - benchmark_monthly[period]),
            }
            for period in sorted(set(portfolio_monthly) & set(benchmark_monthly))
        ]
        annual = [
            {
                "period": period,
                "replication": str(portfolio_annual[period]),
                "benchmark": str(benchmark_annual[period]),
                "difference": str(portfolio_annual[period] - benchmark_annual[period]),
            }
            for period in sorted(set(portfolio_annual) & set(benchmark_annual))
        ]
        tracking_error = _tracking_error(
            portfolio,
            benchmark,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        gate_config = schedule_config["quality_gate"]
        gate_metrics = {
            "maximum_monthly_difference_bps": max(
                abs(float(item["difference"])) for item in monthly
            )
            * 10_000,
            "maximum_annual_difference_bps": max(abs(float(item["difference"])) for item in annual)
            * 10_000,
            "annualized_tracking_error_bps": tracking_error * 10_000,
        }
        gate_passed = all(
            gate_metrics[key] <= float(gate_config[threshold])
            for key, threshold in (
                ("maximum_monthly_difference_bps", "maximum_monthly_difference_bps"),
                ("maximum_annual_difference_bps", "maximum_annual_difference_bps"),
                ("annualized_tracking_error_bps", "maximum_annualized_tracking_error_bps"),
            )
        )
        stable: dict[str, object] = {
            "status": ("PASS_DEBUG_CALIBRATION" if gate_passed else "FAILED_DEBUG_CALIBRATION"),
            "index_code": args.index_code,
            "start_date": args.start_date.isoformat(),
            "end_date": args.end_date.isoformat(),
            "initial_equity": str(args.initial_equity),
            "history_release_id": release_id,
            "history_release_hash": _hash(release_manifest),
            "index_source_fingerprint": index.source_fingerprint,
            "schedule_config_hash": _hash(schedule_config),
            "sessions": len(sessions),
            "constituent_union": len(all_symbols),
            "rebalance_count": len(targets),
            "rebalances": [
                {
                    "signal_date": target.signal_date.isoformat(),
                    "effective_date": target.effective_date.isoformat(),
                    "source_snapshot_date": target.source_snapshot_date.isoformat(),
                    "reason": target.reason,
                    "constituents": len(target.weights),
                }
                for target in targets.values()
            ],
            "orders": len(result.orders),
            "fills": len(result.fills),
            "total_return_from_engine_start": str(metrics.total_return),
            "maximum_drawdown": str(metrics.max_drawdown),
            "tracking_error": tracking_error,
            "quality_gate": {
                "passed": gate_passed,
                "thresholds": gate_config,
                "metrics": gate_metrics,
            },
            "monthly": monthly,
            "annual": annual,
            "final_state_hash": result.final_state_hash,
            "methodology": {
                "cash_dividends": "removed_to_match_price_index",
                "fees": "zero",
                "slippage": "zero",
                "tradability_blocks": "ignored",
                "missing_bar_valuation": "last_visible_close_for_index_calibration",
                "execution": "prior_close_signal_next_open_fill",
                "regular_adjustment": "June/December second Friday next session",
                "temporary_adjustment": "first observed membership-change snapshot",
            },
        }
        content_hash = _hash(stable)
        stable["content_hash"] = content_hash
        payload = {**stable, "created_at": datetime.now(UTC).isoformat(timespec="microseconds")}
        json_path = args.output_dir / f"csi300-replication-{content_hash[:12]}.json"
        markdown_path = args.output_dir / f"csi300-replication-{content_hash[:12]}.md"
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
                    "tracking_error": stable["tracking_error"],
                    "annual": annual,
                    "content_hash": content_hash,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
