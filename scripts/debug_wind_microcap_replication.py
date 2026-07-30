#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import duckdb
import yaml

from aquant.data.daily_update import parse_tushare_symbol
from aquant.data.history import HistoryReleaseReader
from aquant.strategies.microcap import (
    WindMicrocapConstituentReturn,
    WindMicrocapDailyPoint,
    calculate_wind_microcap_daily_equal,
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


def _write_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _fetch_json(
    base_url: str,
    params: dict[str, object],
    *,
    timeout: float = 20,
) -> tuple[dict[str, object], str]:
    url = f"{base_url}?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "aquant/0.3 wind-index-debug",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
    payload = json.loads(body)
    if not isinstance(payload, dict) or payload.get("Success") is not True:
        raise ValueError(f"Wind public index request failed: {base_url}")
    return payload, hashlib.sha256(body).hexdigest()


def _official_benchmark(
    config: dict[str, object],
    *,
    index_code: str,
) -> tuple[dict[date, Decimal], dict[date, Decimal], dict[int, Decimal], str]:
    benchmark = config["benchmark"]
    if not isinstance(benchmark, dict):
        raise ValueError("Wind benchmark configuration is invalid")
    search, search_hash = _fetch_json(
        str(benchmark["search_url"]),
        {"keyWord": index_code, "recordCount": 10, "lan": "cn"},
    )
    search_results = search.get("Result")
    if not isinstance(search_results, list):
        raise ValueError("Wind index search response is invalid")
    matches = [
        item
        for item in search_results
        if isinstance(item, dict) and item.get("windCode") == index_code
    ]
    if len(matches) != 1:
        raise ValueError("Wind index search did not resolve one exact index")
    cipher = str(matches[0]["cipher"])

    history, history_hash = _fetch_json(
        str(benchmark["weekly_history_url"]),
        {"indexid": cipher, "lan": "cn"},
    )
    history_results = history.get("Result")
    if not isinstance(history_results, list):
        raise ValueError("Wind weekly history response is invalid")
    history_matches = [
        item
        for item in history_results
        if isinstance(item, dict) and item.get("windCode") == index_code
    ]
    if len(history_matches) != 1:
        raise ValueError("Wind weekly history has no exact index series")
    series = history_matches[0]
    timestamps = series.get("date")
    closes = series.get("close")
    if (
        not isinstance(timestamps, list)
        or not isinstance(closes, list)
        or len(timestamps) != len(closes)
        or len(closes) < 2
    ):
        raise ValueError("Wind weekly history arrays are invalid")
    weekly = {
        datetime.fromtimestamp(float(timestamp) / 1000, UTC).date(): Decimal(str(close))
        for timestamp, close in zip(timestamps, closes, strict=True)
        if Decimal(str(close)) > 0
    }

    recent, recent_hash = _fetch_json(
        str(benchmark["recent_daily_history_url"]),
        {"indexId": cipher, "period": "1Y", "lan": "cn"},
    )
    recent_result = recent.get("Result")
    if not isinstance(recent_result, dict):
        raise ValueError("Wind recent daily history response is invalid")
    recent_rows = recent_result.get("data")
    if not isinstance(recent_rows, list):
        raise ValueError("Wind recent daily history rows are invalid")
    recent_daily = {
        _date(str(item["tradeDate"])): Decimal(str(item["close"]))
        for item in recent_rows
        if isinstance(item, dict)
        and item.get("windCode") == index_code
        and Decimal(str(item["close"])) > 0
    }

    annual, annual_hash = _fetch_json(
        str(benchmark["annual_return_url"]),
        {"indexid": cipher, "lan": "cn"},
    )
    annual_results = annual.get("Result")
    if not isinstance(annual_results, list):
        raise ValueError("Wind annual return response is invalid")
    annual_matches = [
        item
        for item in annual_results
        if isinstance(item, dict) and item.get("windCode") == index_code
    ]
    if len(annual_matches) != 1:
        raise ValueError("Wind annual return response has no exact index")
    latest_year = max(weekly).year
    annual_returns: dict[int, Decimal] = {}
    for offset in range(1, 10):
        value = annual_matches[0].get(f"year{offset}")
        if value is not None:
            annual_returns[latest_year - offset] = Decimal(str(value)) / Decimal("100")
    fingerprint = hashlib.sha256(
        "".join(sorted((search_hash, history_hash, recent_hash, annual_hash))).encode()
    ).hexdigest()
    return weekly, recent_daily, annual_returns, fingerprint


def _load_constituent_returns(
    release_directory: Path,
    *,
    start_date: date,
    end_date: date,
    target_count: int,
) -> tuple[tuple[WindMicrocapConstituentReturn, ...], str, dict[str, int]]:
    release = HistoryReleaseReader(release_directory)
    release.require(
        "daily",
        "daily_basic",
        "namechange",
        "stock_basic",
        "stk_limit",
        "suspend_d",
        "trade_cal",
    )
    signal_start = start_date - timedelta(days=45)
    price_start = start_date - timedelta(days=730)
    ipo_history_start = start_date - timedelta(days=370)
    sql = """
        WITH calendar AS (
            SELECT
                cal_date,
                row_number() OVER (ORDER BY cal_date) AS session_number
            FROM (
                SELECT DISTINCT cal_date
                FROM read_parquet(?)
                WHERE is_open
                  AND exchange IN ('SSE', 'SZSE')
                  AND cal_date BETWEEN ? AND ?
            )
        ),
        ipo_unlocked AS (
            SELECT daily.ts_code, min(daily.trade_date) AS unlocked_date
            FROM read_parquet(?) AS daily
            JOIN read_parquet(?) AS limits USING (ts_code, trade_date)
            JOIN read_parquet(?) AS instrument USING (ts_code)
            WHERE instrument.list_date >= ?
              AND daily.trade_date <= ?
              AND daily.low < limits.up_limit
            GROUP BY daily.ts_code
        ),
        current_basic AS (
            SELECT trade_date, ts_code, total_market_cap
            FROM read_parquet(?)
            WHERE trade_date BETWEEN ? AND ?
        ),
        suspensions AS (
            SELECT DISTINCT trade_date, ts_code
            FROM read_parquet(?)
            WHERE trade_date BETWEEN ? AND ?
        ),
        carried_suspended AS (
            SELECT
                suspension.trade_date,
                history.ts_code,
                history.total_market_cap
            FROM suspensions AS suspension
            JOIN LATERAL (
                SELECT basic.ts_code, basic.total_market_cap
                FROM read_parquet(?) AS basic
                WHERE basic.ts_code = suspension.ts_code
                  AND basic.trade_date < suspension.trade_date
                ORDER BY basic.trade_date DESC
                LIMIT 1
            ) AS history ON TRUE
            WHERE NOT EXISTS (
                SELECT 1
                FROM current_basic AS current
                WHERE current.trade_date = suspension.trade_date
                  AND current.ts_code = suspension.ts_code
            )
        ),
        basic_asof AS (
            SELECT * FROM current_basic
            UNION ALL
            SELECT * FROM carried_suspended
        ),
        historical_state AS (
            SELECT
                basic.trade_date,
                basic.ts_code,
                basic.total_market_cap,
                instrument.list_date,
                name.name,
                unlock.unlocked_date,
                row_number() OVER (
                    PARTITION BY basic.trade_date, basic.ts_code
                    ORDER BY
                        name.start_date DESC,
                        coalesce(name.end_date, DATE '9999-12-31') DESC
                ) AS name_rank
            FROM basic_asof AS basic
            JOIN read_parquet(?) AS instrument USING (ts_code)
            LEFT JOIN read_parquet(?) AS name
              ON name.ts_code = basic.ts_code
             AND name.start_date <= basic.trade_date
             AND (name.end_date IS NULL OR name.end_date >= basic.trade_date)
            LEFT JOIN ipo_unlocked AS unlock USING (ts_code)
            WHERE (basic.ts_code LIKE '%.SH' OR basic.ts_code LIKE '%.SZ')
              AND instrument.list_date <= basic.trade_date
              AND (
                    instrument.delist_date IS NULL
                    OR instrument.delist_date > basic.trade_date
              )
              AND (
                    instrument.list_date < ?
                    OR unlock.unlocked_date <= basic.trade_date
              )
        ),
        ranked AS (
            SELECT
                trade_date,
                ts_code,
                row_number() OVER (
                    PARTITION BY trade_date
                    ORDER BY total_market_cap, ts_code
                ) AS cap_rank
            FROM historical_state
            WHERE name_rank = 1
              AND name IS NOT NULL
              AND upper(name) NOT LIKE '%ST%'
              AND name NOT LIKE '%退%'
        ),
        selected AS (
            SELECT trade_date, ts_code
            FROM ranked
            WHERE cap_rank <= ?
        ),
        prices AS (
            SELECT
                ts_code,
                trade_date,
                close,
                lag(close) OVER (
                    PARTITION BY ts_code
                    ORDER BY trade_date
                ) AS previous_visible_close
            FROM read_parquet(?)
            WHERE trade_date BETWEEN ? AND ?
        ),
        returns AS (
            SELECT
                ts_code,
                trade_date,
                close / previous_visible_close - 1 AS simple_return
            FROM prices
            WHERE previous_visible_close IS NOT NULL
        )
        SELECT
            effective.cal_date AS trade_date,
            selected.ts_code,
            coalesce(returns.simple_return, 0) AS simple_return
        FROM selected
        JOIN calendar AS signal
          ON signal.cal_date = selected.trade_date
        JOIN calendar AS effective
          ON effective.session_number = signal.session_number + 1
        LEFT JOIN returns
          ON returns.ts_code = selected.ts_code
         AND returns.trade_date = effective.cal_date
        WHERE effective.cal_date BETWEEN ? AND ?
        ORDER BY effective.cal_date, selected.ts_code
    """
    connection = duckdb.connect(":memory:")
    try:
        rows = connection.execute(
            sql,
            [
                release.parquet_pattern("trade_cal"),
                signal_start,
                end_date,
                release.parquet_pattern("daily"),
                release.parquet_pattern("stk_limit"),
                release.parquet_pattern("stock_basic"),
                ipo_history_start,
                end_date,
                release.parquet_pattern("daily_basic"),
                signal_start,
                end_date,
                release.parquet_pattern("suspend_d"),
                signal_start,
                end_date,
                release.parquet_pattern("daily_basic"),
                release.parquet_pattern("stock_basic"),
                release.parquet_pattern("namechange"),
                ipo_history_start,
                target_count,
                release.parquet_pattern("daily"),
                price_start,
                end_date,
                start_date,
                end_date,
            ],
        ).fetchall()
    finally:
        connection.close()
    observations = tuple(
        WindMicrocapConstituentReturn(
            trade_date=row[0],
            symbol=parse_tushare_symbol(row[1]),
            simple_return=Decimal(str(row[2])),
        )
        for row in rows
    )
    counts: dict[date, int] = defaultdict(int)
    source_hasher = hashlib.sha256()
    zero_return_count = 0
    for observation in observations:
        counts[observation.trade_date] += 1
        zero_return_count += observation.simple_return == 0
        source_hasher.update(
            (
                f"{observation.trade_date.isoformat()}|"
                f"{observation.symbol.canonical}|{observation.simple_return}\n"
            ).encode()
        )
    incomplete = {day: count for day, count in counts.items() if count != target_count}
    if incomplete:
        first = min(incomplete)
        raise ValueError(
            f"Wind microcap selection is incomplete on {first}: "
            f"{incomplete[first]} != {target_count}"
        )
    return (
        observations,
        source_hasher.hexdigest(),
        {
            "sessions": len(counts),
            "rows": len(observations),
            "zero_constituent_returns": zero_return_count,
        },
    )


def _period_returns(
    values: dict[date, Decimal],
    *,
    annual: bool,
    initial_value: Decimal | None = Decimal("1"),
) -> dict[str, Decimal]:
    if not values:
        raise ValueError("return series must not be empty")
    previous = initial_value
    ends: dict[str, Decimal] = {}
    for day, value in sorted(values.items()):
        key = str(day.year) if annual else day.strftime("%Y-%m")
        ends[key] = value
    output: dict[str, Decimal] = {}
    for key, ending in sorted(ends.items()):
        if previous is not None:
            output[key] = ending / previous - Decimal("1")
        previous = ending
    return output


def _benchmark_period_returns(
    closes: dict[date, Decimal],
    *,
    start_date: date,
    end_date: date,
    annual: bool,
) -> dict[str, Decimal]:
    prior = [value for day, value in sorted(closes.items()) if day < start_date]
    if not prior:
        raise ValueError("Wind benchmark has no observation before the requested period")
    relevant = {day: value for day, value in closes.items() if start_date <= day <= end_date}
    return _period_returns(relevant, annual=annual, initial_value=prior[-1])


def _monthly_turnover(points: tuple[WindMicrocapDailyPoint, ...]) -> dict[str, Decimal]:
    output: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for point in points:
        output[point.trade_date.strftime("%Y-%m")] += point.two_way_turnover
    return dict(output)


def _exact_monthly_benchmark(
    closes: dict[date, Decimal],
    strategy_values: dict[date, Decimal],
) -> dict[str, Decimal]:
    strategy_ends: dict[str, date] = {}
    for day in sorted(strategy_values):
        strategy_ends[day.strftime("%Y-%m")] = day
    ordered_ends = sorted(strategy_ends.items())
    output: dict[str, Decimal] = {}
    for index in range(1, len(ordered_ends)):
        period, ending_day = ordered_ends[index]
        _prior_period, prior_day = ordered_ends[index - 1]
        if ending_day in closes and prior_day in closes:
            output[period] = closes[ending_day] / closes[prior_day] - Decimal("1")
    return output


def _markdown(payload: dict[str, object]) -> str:
    monthly = payload["monthly"]
    annual = payload["annual"]
    assert isinstance(monthly, list)
    assert isinstance(annual, list)
    lines = [
        "# Wind microcap daily equal-weight replication debug",
        "",
        f"- Period: {payload['start_date']} to {payload['end_date']}",
        f"- Index: `{payload['index_code']}`",
        f"- Status: `{payload['status']}`",
        f"- Constituents per session: {payload['target_count']}",
        "",
        "| Month | Replication | Wind benchmark | Sampling | Difference | "
        "Two-way turnover | Friction |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in monthly:
        assert isinstance(row, dict)
        lines.append(
            f"| {row['period']} | {Decimal(str(row['replication'])):.4%} "
            f"| {Decimal(str(row['benchmark'])):.4%} "
            f"| {row['benchmark_sampling']} "
            f"| {Decimal(str(row['difference'])):.4%} "
            f"| {Decimal(str(row['two_way_turnover'])):.4f} | 0.0000% |"
        )
    lines.extend(
        (
            "",
            "| Year | Replication | Wind official | Difference |",
            "| --- | ---: | ---: | ---: |",
        )
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
            "Conventions: prior-close PIT selection, smallest 400 by total market cap, "
            "daily arithmetic equal weighting, no fees or slippage, and no price-limit "
            "execution blocking. Suspended names retain their last visible market cap "
            "and contribute zero until the next visible close.",
            "",
            "The public Wind long-history chart is weekly sampled, so monthly benchmark "
            "returns are diagnostic approximations. Annual benchmark returns come from "
            "Wind's official annual-return endpoint.",
            "",
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replicate the historical Wind microcap daily equal-weight index"
    )
    parser.add_argument("--history-release", type=Path, required=True)
    parser.add_argument("--start-date", type=_date, default=date(2024, 1, 1))
    parser.add_argument("--end-date", type=_date, default=date(2025, 12, 31))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/index_replication/wind_microcap_daily_equal_2024_2025.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/backtest_debug/wind_microcap"),
    )
    args = parser.parse_args(argv)
    try:
        if args.start_date > args.end_date:
            raise ValueError("replication date range is inverted")
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        index_code = str(config["index_code"])
        target_count = int(config["target_count"])
        observations, selection_hash, data_stats = _load_constituent_returns(
            args.history_release,
            start_date=args.start_date,
            end_date=args.end_date,
            target_count=target_count,
        )
        points = calculate_wind_microcap_daily_equal(
            observations,
            target_count=target_count,
        )
        (
            weekly_benchmark,
            recent_daily_benchmark,
            official_annual,
            benchmark_hash,
        ) = _official_benchmark(config, index_code=index_code)
        nav = {point.trade_date: point.net_value for point in points}
        replication_monthly = _period_returns(nav, annual=False)
        replication_annual = _period_returns(nav, annual=True)
        benchmark_monthly = _benchmark_period_returns(
            weekly_benchmark,
            start_date=args.start_date,
            end_date=args.end_date,
            annual=False,
        )
        exact_monthly = _exact_monthly_benchmark(recent_daily_benchmark, nav)
        turnover = _monthly_turnover(points)
        monthly = [
            {
                "period": period,
                "replication": str(replication_monthly[period]),
                "benchmark": str(exact_monthly.get(period, benchmark_monthly[period])),
                "difference": str(
                    replication_monthly[period]
                    - exact_monthly.get(period, benchmark_monthly[period])
                ),
                "benchmark_sampling": (
                    "official_daily" if period in exact_monthly else "official_weekly_sample"
                ),
                "two_way_turnover": str(turnover[period]),
                "friction_cost": "0",
            }
            for period in sorted(set(replication_monthly) & set(benchmark_monthly))
        ]
        annual = [
            {
                "period": period,
                "replication": str(replication_annual[period]),
                "benchmark": str(official_annual[int(period)]),
                "difference": str(replication_annual[period] - official_annual[int(period)]),
            }
            for period in sorted(replication_annual)
            if int(period) in official_annual
        ]
        if not monthly or not annual:
            raise ValueError("Wind benchmark has insufficient overlap")
        gates = config["quality_gate"]
        exact_monthly_differences = [
            abs(Decimal(str(item["difference"])))
            for item in monthly
            if item["benchmark_sampling"] == "official_daily"
        ]
        if not exact_monthly_differences:
            raise ValueError("Wind benchmark has no exact daily monthly overlap")
        max_monthly = max(exact_monthly_differences)
        max_annual = max(abs(Decimal(str(item["difference"]))) for item in annual)
        gate_passed = max_monthly <= Decimal(
            str(gates["maximum_monthly_difference_percentage_points"])
        ) / Decimal("100") and max_annual <= Decimal(
            str(gates["maximum_annual_difference_percentage_points"])
        ) / Decimal("100")
        release_manifest = json.loads(
            (args.history_release / "manifest.json").read_text(encoding="utf-8")
        )
        stable: dict[str, object] = {
            "status": ("PASS_DEBUG_CALIBRATION" if gate_passed else "FAILED_DEBUG_CALIBRATION"),
            "index_code": index_code,
            "start_date": args.start_date.isoformat(),
            "end_date": args.end_date.isoformat(),
            "target_count": target_count,
            "history_release_id": release_manifest["release_id"],
            "history_release_hash": _hash(release_manifest),
            "config_hash": _hash(config),
            "selection_hash": selection_hash,
            "benchmark_source_hash": benchmark_hash,
            "benchmark_monthly_sampling": (
                "official daily where fully covered; otherwise official weekly chart"
            ),
            "benchmark_annual_sampling": "official_annual_return_endpoint",
            "data_stats": data_stats,
            "monthly": monthly,
            "annual": annual,
            "total_return": str(points[-1].net_value - Decimal("1")),
            "total_two_way_turnover": str(
                sum((point.two_way_turnover for point in points), Decimal("0"))
            ),
            "friction_cost": "0",
            "quality_gate": {
                "passed": gate_passed,
                "maximum_monthly_difference_percentage_points": str(max_monthly * Decimal("100")),
                "maximum_annual_difference_percentage_points": str(max_annual * Decimal("100")),
                "thresholds": gates,
            },
            "methodology": config,
        }
        content_hash = _hash(stable)
        payload = {
            **stable,
            "content_hash": content_hash,
            "created_at": datetime.now(UTC).isoformat(),
        }
        json_path = args.output_dir / f"wind-microcap-replication-{content_hash[:12]}.json"
        markdown_path = json_path.with_suffix(".md")
        _write_atomic(
            json_path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        )
        _write_atomic(markdown_path, _markdown(payload))
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "report": str(json_path),
                    "content_hash": content_hash,
                    "annual": annual,
                    "quality_gate": payload["quality_gate"],
                    "sessions": data_stats["sessions"],
                    "rows": data_stats["rows"],
                    "total_two_way_turnover": payload["total_two_way_turnover"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if gate_passed else 2
    except (KeyError, OSError, TypeError, ValueError, duckdb.Error) as exc:
        parser.exit(1, f"wind microcap replication failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
