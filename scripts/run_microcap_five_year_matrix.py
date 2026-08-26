#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import resource
import time
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import duckdb

from aquant.backtest import (
    EventDrivenBacktest,
    MarketSession,
    UnfilledOrderPolicy,
    calculate_metrics,
)
from aquant.data.daily_update import parse_tushare_symbol
from aquant.data.history import DuckDBMicrocapHistory, HistoryReleaseReader
from aquant.domain.identifiers import Symbol
from aquant.strategies.microcap import (
    PROTOTYPE_DEFINITIONS,
    CostScenario,
    MicrocapPrototype,
    MicrocapRankBandStrategy,
    MicrocapRankBandTarget,
    RebalanceFrequency,
    generate_rebalance_dates,
    matcher_for_cost_scenario,
)
from aquant.strategies.quantile_report import monthly_portfolio_results

DEFAULT_RELEASE = Path("data/standard/history-release=cn_equity_history_20260717_001")
DEFAULT_START = date(2021, 7, 19)
DEFAULT_END = date(2026, 7, 17)
DEFAULT_CAPITAL = Decimal("10000000")


def _date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYYMMDD") from exc


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _without_runtime_metrics(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_runtime_metrics(item)
            for key, item in value.items()
            if key not in {"elapsed_seconds", "peak_rss_mib"}
        }
    if isinstance(value, list):
        return [_without_runtime_metrics(item) for item in value]
    return value


def _write_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _build_eligible_table(
    connection: duckdb.DuckDBPyConnection,
    release: HistoryReleaseReader,
    *,
    start_date: date,
    end_date: date,
) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE eligible AS
        WITH financial_deduplicated AS (
            SELECT * EXCLUDE (revision_rank)
            FROM (
                SELECT
                    ts_code,
                    ann_date,
                    end_date,
                    netprofit_yoy,
                    debt_to_assets,
                    row_number() OVER (
                        PARTITION BY ts_code, ann_date
                        ORDER BY
                            end_date DESC,
                            coalesce(update_flag, '') DESC,
                            netprofit_yoy DESC NULLS LAST,
                            debt_to_assets DESC NULLS LAST
                    ) AS revision_rank
                FROM read_parquet(?)
            )
            WHERE revision_rank = 1
        ),
        basic_with_financial AS (
            SELECT
                basic.ts_code,
                basic.trade_date,
                basic.total_market_cap,
                basic.float_market_cap,
                basic.pb,
                basic.dividend_yield,
                basic.turnover_volatility_20d,
                financial.netprofit_yoy,
                financial.debt_to_assets
            FROM (
                SELECT basic.*
                FROM read_parquet(?) AS basic
                JOIN read_parquet(?) AS bar USING (ts_code, trade_date)
                WHERE basic.trade_date BETWEEN ? AND ?
            ) AS basic
            ASOF LEFT JOIN financial_deduplicated AS financial
              ON basic.ts_code = financial.ts_code
             AND basic.trade_date > financial.ann_date
        ),
        historical_state AS (
            SELECT
                basic.*,
                name.name,
                row_number() OVER (
                    PARTITION BY basic.trade_date, basic.ts_code
                    ORDER BY
                        name.start_date DESC,
                        coalesce(name.end_date, DATE '9999-12-31') DESC,
                        name.name DESC
                ) AS name_rank
            FROM basic_with_financial AS basic
            JOIN read_parquet(?) AS instrument USING (ts_code)
            LEFT JOIN read_parquet(?) AS name
              ON name.ts_code = basic.ts_code
             AND name.start_date <= basic.trade_date
             AND (name.end_date IS NULL OR name.end_date >= basic.trade_date)
            WHERE (basic.ts_code LIKE '%.SH' OR basic.ts_code LIKE '%.SZ')
              AND instrument.list_date <= basic.trade_date - INTERVAL 120 DAY
              AND (
                    instrument.delist_date IS NULL
                    OR instrument.delist_date > basic.trade_date
              )
              AND basic.total_market_cap > 0
        )
        SELECT
            trade_date,
            ts_code,
            total_market_cap,
            float_market_cap,
            pb,
            dividend_yield,
            turnover_volatility_20d,
            netprofit_yoy,
            debt_to_assets,
            row_number() OVER (
                PARTITION BY trade_date
                ORDER BY total_market_cap, ts_code
            ) AS cap_rank
        FROM historical_state
        WHERE name_rank = 1
          AND name IS NOT NULL
          AND upper(name) NOT LIKE '%ST%'
          AND name NOT LIKE '%é€€%'
        """,
        [
            release.parquet_pattern("fina_indicator"),
            release.parquet_pattern("daily_basic"),
            release.parquet_pattern("daily"),
            start_date,
            end_date,
            release.parquet_pattern("stock_basic"),
            release.parquet_pattern("namechange"),
        ],
    )


def _target_queries() -> dict[MicrocapPrototype, str]:
    return {
        MicrocapPrototype.SMALLEST_100: """
            SELECT trade_date, cap_rank, ts_code
            FROM eligible WHERE cap_rank <= 100
            ORDER BY trade_date, cap_rank
        """,
        MicrocapPrototype.RANK_101_400: """
            SELECT trade_date, cap_rank, ts_code
            FROM eligible WHERE cap_rank BETWEEN 101 AND 400
            ORDER BY trade_date, cap_rank
        """,
        MicrocapPrototype.SMALLEST_400: """
            SELECT trade_date, cap_rank, ts_code
            FROM eligible WHERE cap_rank <= 400
            ORDER BY trade_date, cap_rank
        """,
        MicrocapPrototype.EXECUTABLE_95: """
            SELECT trade_date, candidate_rank, ts_code
            FROM (
                SELECT
                    trade_date,
                    ts_code,
                    row_number() OVER (
                        PARTITION BY trade_date
                        ORDER BY total_market_cap, ts_code
                    ) AS candidate_rank
                FROM eligible
                WHERE pb > 0 AND netprofit_yoy > 0
            )
            WHERE candidate_rank <= 120
            ORDER BY trade_date, candidate_rank
        """,
        MicrocapPrototype.DIVIDEND_QUALITY_10: """
            WITH complete AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY trade_date
                        ORDER BY dividend_yield DESC, ts_code DESC
                    ) AS factor_rank,
                    count(*) OVER (PARTITION BY trade_date) AS factor_count
                FROM eligible
                WHERE dividend_yield IS NOT NULL
                  AND debt_to_assets IS NOT NULL
                  AND turnover_volatility_20d IS NOT NULL
            ),
            high_dividend AS (
                SELECT *
                FROM complete
                WHERE factor_rank <= ceil(factor_count / 2.0)
            ),
            leverage_ranked AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY trade_date
                        ORDER BY debt_to_assets, ts_code
                    ) AS leverage_rank,
                    count(*) OVER (PARTITION BY trade_date) AS leverage_count
                FROM high_dividend
            ),
            low_leverage AS (
                SELECT *
                FROM leverage_ranked
                WHERE leverage_rank <= ceil(leverage_count / 2.0)
            ),
            turnover_ranked AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY trade_date
                        ORDER BY turnover_volatility_20d, ts_code
                    ) AS turnover_rank,
                    count(*) OVER (PARTITION BY trade_date) AS turnover_count
                FROM low_leverage
            ),
            low_turnover AS (
                SELECT *
                FROM turnover_ranked
                WHERE turnover_rank <= ceil(turnover_count / 2.0)
            ),
            selected AS (
                SELECT
                    trade_date,
                    ts_code,
                    row_number() OVER (
                        PARTITION BY trade_date
                        ORDER BY coalesce(float_market_cap, total_market_cap), ts_code
                    ) AS candidate_rank
                FROM low_turnover
            )
            SELECT trade_date, candidate_rank, ts_code
            FROM selected
            WHERE candidate_rank <= 10
            ORDER BY trade_date, candidate_rank
        """,
        MicrocapPrototype.LOW_PB_LOW_TURNOVER_35: """
            WITH complete AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY trade_date
                        ORDER BY pb, ts_code
                    ) AS factor_rank,
                    count(*) OVER (PARTITION BY trade_date) AS factor_count
                FROM eligible
                WHERE pb > 0 AND turnover_volatility_20d IS NOT NULL
            ),
            low_pb AS (
                SELECT *
                FROM complete
                WHERE factor_rank <= ceil(factor_count / 2.0)
            ),
            turnover_ranked AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY trade_date
                        ORDER BY turnover_volatility_20d, ts_code
                    ) AS turnover_rank,
                    count(*) OVER (PARTITION BY trade_date) AS turnover_count
                FROM low_pb
            ),
            low_turnover AS (
                SELECT *
                FROM turnover_ranked
                WHERE turnover_rank <= ceil(turnover_count / 2.0)
            ),
            selected AS (
                SELECT
                    trade_date,
                    ts_code,
                    row_number() OVER (
                        PARTITION BY trade_date
                        ORDER BY total_market_cap, ts_code
                    ) AS candidate_rank
                FROM low_turnover
            )
            SELECT trade_date, candidate_rank, ts_code
            FROM selected
            WHERE candidate_rank <= 35
            ORDER BY trade_date, candidate_rank
        """,
    }


def _load_targets(
    release_directory: Path,
    *,
    start_date: date,
    end_date: date,
    trading_dates: Sequence[date],
) -> tuple[
    dict[MicrocapPrototype, dict[date, MicrocapRankBandTarget]],
    str,
]:
    release = HistoryReleaseReader(release_directory)
    release.require(
        "daily",
        "daily_basic",
        "fina_indicator",
        "namechange",
        "stock_basic",
    )
    connection = duckdb.connect(":memory:")
    connection.execute("SET memory_limit = '4GB'")
    try:
        _build_eligible_table(
            connection,
            release,
            start_date=start_date,
            end_date=end_date,
        )
        targets: dict[MicrocapPrototype, dict[date, MicrocapRankBandTarget]] = {}
        source_hasher = hashlib.sha256()
        expected_dates = set(trading_dates)
        for prototype, sql in _target_queries().items():
            grouped: dict[date, list[Symbol]] = {}
            for trade_date, _rank, ts_code in connection.execute(sql).fetchall():
                grouped.setdefault(trade_date, []).append(parse_tushare_symbol(ts_code))
            missing_dates = expected_dates - set(grouped)
            if missing_dates:
                raise ValueError(f"{prototype.value} has no candidates on {min(missing_dates)}")
            definition = PROTOTYPE_DEFINITIONS[prototype]
            expected_candidates = definition.sell_buffer_rank or definition.target_count
            incomplete = {
                day: len(symbols)
                for day, symbols in grouped.items()
                if len(symbols) != expected_candidates
            }
            if incomplete:
                first = min(incomplete)
                raise ValueError(
                    f"{prototype.value} target is incomplete on {first}: "
                    f"{incomplete[first]} != {expected_candidates}"
                )
            entries: dict[date, MicrocapRankBandTarget] = {}
            for trade_date, symbols in sorted(grouped.items()):
                immutable = tuple(symbols)
                entries[trade_date] = MicrocapRankBandTarget(
                    trade_date,
                    definition.rank_start + 1,
                    definition.rank_start + len(immutable),
                    immutable,
                    target_count=definition.target_count,
                    sell_buffer_rank=definition.sell_buffer_rank,
                )
                source_hasher.update(
                    (
                        f"{prototype.value}|{trade_date.isoformat()}|"
                        + ",".join(symbol.canonical for symbol in immutable)
                        + "\n"
                    ).encode()
                )
            targets[prototype] = entries
    finally:
        connection.close()
    return targets, source_hasher.hexdigest()


def _slippage(
    sessions: Sequence[MarketSession],
    fills: Sequence[object],
) -> Decimal:
    session_by_date = {session.trade_date: session for session in sessions}
    total = Decimal("0")
    for fill in fills:
        occurred_at = fill.occurred_at
        symbol = fill.symbol
        price = fill.price
        quantity = fill.quantity
        reference = session_by_date[occurred_at.date()].bar_by_symbol()[symbol].open
        total += abs(price - reference) * quantity
    return total


def _annual_returns(
    equity_curve: Sequence[object],
    *,
    initial_equity: Decimal,
) -> dict[str, Decimal]:
    endings: dict[str, Decimal] = {}
    for snapshot in equity_curve:
        endings[str(snapshot.asof_time.year)] = snapshot.equity
    previous = initial_equity
    output: dict[str, Decimal] = {}
    for year, ending in sorted(endings.items()):
        output[year] = ending / previous - Decimal("1")
        previous = ending
    return output


def _run_one(
    *,
    prototype: MicrocapPrototype,
    frequency: RebalanceFrequency,
    scenario: CostScenario,
    capital: Decimal,
    sessions: tuple[MarketSession, ...],
    all_targets: Mapping[date, MicrocapRankBandTarget],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, Decimal]]:
    signal_dates = generate_rebalance_dates(
        tuple(session.trade_date for session in sessions),
        frequency,
    )
    strategy = MicrocapRankBandStrategy(
        targets={day: all_targets[day] for day in signal_dates},
    )
    started = time.perf_counter()
    result = EventDrivenBacktest(
        run_id=f"microcap-5y-{prototype.value}-{frequency.value}-{scenario.value}",
        initial_cash=capital,
        matcher=matcher_for_cost_scenario(scenario),
        unfilled_order_policy=UnfilledOrderPolicy.CANCEL_AFTER_OPEN,
    ).run(sessions, strategy)
    elapsed = time.perf_counter() - started
    metrics = calculate_metrics(result, initial_equity=capital)
    slip = _slippage(sessions, result.fills)
    monthly = [
        row.as_dict()
        for row in monthly_portfolio_results(
            result,
            initial_equity=capital,
            sessions=sessions,
        )
    ]
    holdings = [len(snapshot.positions) for snapshot in result.equity_curve]
    diagnostics = strategy.diagnostics
    summary: dict[str, object] = {
        "scenario": scenario.value,
        "total_return": str(metrics.total_return),
        "final_equity": str(metrics.final_equity),
        "max_drawdown": str(metrics.max_drawdown),
        "gross_turnover": str(metrics.gross_turnover),
        "fees": str(metrics.total_fees),
        "slippage": str(slip),
        "friction_cost": str(metrics.total_fees + slip),
        "fill_rate": str(metrics.fill_rate),
        "trade_count": metrics.trade_count,
        "average_holdings": str(Decimal(sum(holdings)) / len(holdings)),
        "minimum_holdings": min(holdings),
        "maximum_holdings": max(holdings),
        "average_zero_lot_targets": str(
            Decimal(sum(item.zero_lot_count for item in diagnostics)) / len(diagnostics)
        ),
        "average_unpriced_targets": str(
            Decimal(sum(item.target_count - item.priced_count for item in diagnostics))
            / len(diagnostics)
        ),
        "elapsed_seconds": elapsed,
        "final_state_hash": result.final_state_hash,
    }
    return (
        summary,
        monthly,
        _annual_returns(
            result.equity_curve,
            initial_equity=capital,
        ),
    )


def _markdown(payload: dict[str, object]) -> str:
    summaries = payload["summaries"]
    annual = payload["annual"]
    assert isinstance(summaries, list)
    assert isinstance(annual, list)
    lines = [
        "# Five-year micro-cap experiment matrix",
        "",
        f"- Period: {payload['start_date']} to {payload['end_date']}",
        f"- Initial capital per run: RMB {Decimal(str(payload['initial_capital'])):,.2f}",
        f"- Data release: `{payload['data_release_id']}`",
        "- Comparison: identical A-share execution rules; zero-cost has no fees/slippage, "
        "base-cost adds the configured fee schedule and 5 bps slippage.",
        "",
        "| Prototype | Rebalance | Zero-cost return | Base-cost return | Return drag "
        "| Zero turnover | Base turnover | Base friction | Max DD (base) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        assert isinstance(row, dict)
        lines.append(
            f"| {row['prototype']} | {row['frequency']} "
            f"| {Decimal(str(row['zero_cost_return'])):.2%} "
            f"| {Decimal(str(row['base_cost_return'])):.2%} "
            f"| {Decimal(str(row['return_drag'])):.2%} "
            f"| {Decimal(str(row['zero_cost_turnover'])):.2f} "
            f"| {Decimal(str(row['base_cost_turnover'])):.2f} "
            f"| {Decimal(str(row['base_friction_cost'])):,.2f} "
            f"| {Decimal(str(row['base_max_drawdown'])):.2%} |"
        )
    lines.extend(
        (
            "",
            "## Annual paired returns",
            "",
            "| Prototype | Rebalance | Year | Zero-cost | Base-cost | Drag |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        )
    )
    for row in annual:
        assert isinstance(row, dict)
        lines.append(
            f"| {row['prototype']} | {row['frequency']} | {row['year']} "
            f"| {Decimal(str(row['zero_cost_return'])):.2%} "
            f"| {Decimal(str(row['base_cost_return'])):.2%} "
            f"| {Decimal(str(row['return_drag'])):.2%} |"
        )
    lines.extend(
        (
            "",
            "Monthly paired returns, turnover, fees and slippage are retained in the "
            "companion JSON artifact.",
            "",
            "Signals use information available after T close and execute at T+1 open. "
            "Selections exclude Beijing exchange, listings younger than 120 calendar days, "
            "historical ST/delisting-risk names, and sessions without a bar. Portfolios are "
            "equal-weighted subject to 100-share board lots and a 2% cash buffer.",
            "",
            "Unfilled and partially filled target differences are retried on subsequent "
            "sessions without recomputing target membership or weights. Realized distinct "
            "holdings can exceed the target count while capacity- or limit-blocked exits "
            "remain in the account; average/minimum/maximum holdings and fill rates are "
            "recorded in JSON as capacity diagnostics.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paired five-year micro-cap matrix")
    parser.add_argument("--release-directory", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--start-date", type=_date, default=DEFAULT_START)
    parser.add_argument("--end-date", type=_date, default=DEFAULT_END)
    parser.add_argument("--initial-capital", type=Decimal, default=DEFAULT_CAPITAL)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/microcap_five_year_matrix"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.start_date >= args.end_date:
        parser.error("start-date must precede end-date")
    if args.initial_capital <= 0:
        parser.error("initial-capital must be positive")

    release = HistoryReleaseReader(args.release_directory)
    overall_started = time.perf_counter()
    with DuckDBMicrocapHistory(args.release_directory) as history:
        trading_dates = history.trading_dates(args.start_date, args.end_date)
    if len(trading_dates) < 2:
        raise ValueError("five-year matrix requires at least two trading sessions")
    print(f"targets: building for {len(trading_dates)} sessions", flush=True)
    targets, target_hash = _load_targets(
        args.release_directory,
        start_date=args.start_date,
        end_date=args.end_date,
        trading_dates=trading_dates,
    )
    unique_codes = tuple(
        sorted(
            {
                symbol.canonical.replace(".XSHG", ".SH").replace(".XSHE", ".SZ")
                for prototype_targets in targets.values()
                for target in prototype_targets.values()
                for symbol in target.symbols
            }
        )
    )
    if args.dry_run:
        print(f"dry-run PASS: target_hash={target_hash} unique_symbols={len(unique_codes)}")
        return 0

    print(f"sessions: loading {len(unique_codes)} selected symbols", flush=True)
    with DuckDBMicrocapHistory(args.release_directory) as history:
        sessions = history.sessions(
            args.start_date,
            args.end_date,
            ts_codes=unique_codes,
        )

    summaries: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    annual_rows: list[dict[str, object]] = []
    for prototype in MicrocapPrototype:
        for frequency in RebalanceFrequency:
            paired: dict[CostScenario, dict[str, object]] = {}
            paired_annual: dict[CostScenario, dict[str, Decimal]] = {}
            paired_monthly: dict[CostScenario, list[dict[str, object]]] = {}
            for scenario in (CostScenario.RULES_ONLY, CostScenario.BASE_COST):
                print(
                    f"run: {prototype.value}/{frequency.value}/{scenario.value}",
                    flush=True,
                )
                summary, monthly, annual = _run_one(
                    prototype=prototype,
                    frequency=frequency,
                    scenario=scenario,
                    capital=args.initial_capital,
                    sessions=sessions,
                    all_targets=targets[prototype],
                )
                paired[scenario] = summary
                paired_monthly[scenario] = monthly
                paired_annual[scenario] = annual
            zero = paired[CostScenario.RULES_ONLY]
            base = paired[CostScenario.BASE_COST]
            summaries.append(
                {
                    "prototype": prototype.value,
                    "frequency": frequency.value,
                    "zero_cost_return": zero["total_return"],
                    "base_cost_return": base["total_return"],
                    "return_drag": str(
                        Decimal(str(zero["total_return"])) - Decimal(str(base["total_return"]))
                    ),
                    "zero_cost_turnover": zero["gross_turnover"],
                    "base_cost_turnover": base["gross_turnover"],
                    "base_friction_cost": base["friction_cost"],
                    "base_fees": base["fees"],
                    "base_slippage": base["slippage"],
                    "base_max_drawdown": base["max_drawdown"],
                    "zero_cost": zero,
                    "base_cost": base,
                }
            )
            zero_monthly = {
                str(row["month"]): row for row in paired_monthly[CostScenario.RULES_ONLY]
            }
            base_monthly = {
                str(row["month"]): row for row in paired_monthly[CostScenario.BASE_COST]
            }
            if set(zero_monthly) != set(base_monthly):
                raise ValueError("paired monthly result periods disagree")
            for month in sorted(zero_monthly):
                zero_row = zero_monthly[month]
                base_row = base_monthly[month]
                monthly_rows.append(
                    {
                        "prototype": prototype.value,
                        "frequency": frequency.value,
                        "month": month,
                        "zero_cost_return": zero_row["net_return"],
                        "base_cost_return": base_row["net_return"],
                        "return_drag": str(
                            Decimal(str(zero_row["net_return"]))
                            - Decimal(str(base_row["net_return"]))
                        ),
                        "zero_cost_turnover": zero_row["turnover"],
                        "base_cost_turnover": base_row["turnover"],
                        "base_fees": base_row["fees"],
                        "base_slippage": base_row["slippage"],
                        "base_friction_cost": base_row["friction_cost"],
                    }
                )
            years = set(paired_annual[CostScenario.RULES_ONLY])
            if years != set(paired_annual[CostScenario.BASE_COST]):
                raise ValueError("paired annual result periods disagree")
            for year in sorted(years):
                zero_return = paired_annual[CostScenario.RULES_ONLY][year]
                base_return = paired_annual[CostScenario.BASE_COST][year]
                annual_rows.append(
                    {
                        "prototype": prototype.value,
                        "frequency": frequency.value,
                        "year": year,
                        "zero_cost_return": str(zero_return),
                        "base_cost_return": str(base_return),
                        "return_drag": str(zero_return - base_return),
                    }
                )

    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "initial_capital": str(args.initial_capital),
        "data_release_id": release.manifest.release_id,
        "target_hash": target_hash,
        "trading_sessions": len(sessions),
        "unique_symbols_loaded": len(unique_codes),
        "strategy_count": len(summaries),
        "paired_run_count": len(summaries) * 2,
        "summaries": summaries,
        "annual": annual_rows,
        "monthly": monthly_rows,
        "elapsed_seconds": time.perf_counter() - overall_started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    payload["content_hash"] = _stable_hash(_without_runtime_metrics(payload))
    stem = f"microcap-5y-matrix-{str(payload['content_hash'])[:12]}"
    json_path = args.output_directory / f"{stem}.json"
    markdown_path = args.output_directory / f"{stem}.md"
    _write_atomic(json_path, json.dumps(payload, ensure_ascii=False, indent=2))
    _write_atomic(markdown_path, _markdown(payload))
    print(
        f"PASS: {len(summaries)} strategies/{len(summaries) * 2} paired runs; "
        f"json={json_path}; report={markdown_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
