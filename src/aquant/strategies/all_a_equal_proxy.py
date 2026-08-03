from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import duckdb

from aquant.data.history.microcap import HistoryReleaseReader


@dataclass(frozen=True, slots=True)
class EqualWeightProxyPoint:
    trade_date: date
    return_rate: float
    constituent_count: int


def build_all_a_equal_weight_proxy(
    release_directory: Path, *, start_date: date, end_date: date
) -> tuple[tuple[EqualWeightProxyPoint, ...], dict[str, object]]:
    """Build an adjusted-close, daily rebalanced all-A equal-weight proxy."""
    if start_date >= end_date:
        raise ValueError("equal-weight proxy dates must be ordered")
    release = HistoryReleaseReader(release_directory)
    release.require("daily", "adj_factor", "stock_basic")
    connection = duckdb.connect(":memory:")
    try:
        rows = connection.execute(
            """
            WITH calendar AS (
                SELECT DISTINCT trade_date
                FROM read_parquet(?)
                WHERE trade_date BETWEEN (
                    SELECT MAX(trade_date)
                    FROM read_parquet(?)
                    WHERE trade_date < ?
                      AND exchange IN ('XSHG', 'XSHE')
                ) AND ?
                  AND exchange IN ('XSHG', 'XSHE')
            ),
            eligible AS (
                SELECT calendar.trade_date, stock.ts_code
                FROM calendar
                JOIN read_parquet(?) AS stock
                  ON stock.list_date <= calendar.trade_date
                 AND (stock.delist_date IS NULL OR stock.delist_date >= calendar.trade_date)
                 AND stock.exchange IN ('XSHG', 'XSHE')
            ),
            prices AS (
                SELECT
                    daily.trade_date,
                    daily.ts_code,
                    CAST(daily.close AS DOUBLE) * CAST(adj.adj_factor AS DOUBLE) AS adjusted_close
                FROM read_parquet(?) AS daily
                JOIN read_parquet(?) AS adj
                  ON adj.trade_date = daily.trade_date
                 AND adj.ts_code = daily.ts_code
                WHERE daily.trade_date <= ?
                  AND daily.exchange IN ('XSHG', 'XSHE')
            ),
            filled AS (
                SELECT
                    eligible.trade_date,
                    eligible.ts_code,
                    LAST_VALUE(prices.adjusted_close IGNORE NULLS) OVER (
                        PARTITION BY eligible.ts_code
                        ORDER BY eligible.trade_date
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS adjusted_close
                FROM eligible
                LEFT JOIN prices
                  ON prices.trade_date = eligible.trade_date
                 AND prices.ts_code = eligible.ts_code
            ),
            returns AS (
                SELECT
                    trade_date,
                    ts_code,
                    adjusted_close / LAG(adjusted_close) OVER (
                        PARTITION BY ts_code ORDER BY trade_date
                    ) - 1 AS return_rate
                FROM filled
            )
            SELECT
                trade_date,
                AVG(COALESCE(return_rate, 0)) AS return_rate,
                COUNT(*) AS constituent_count
            FROM returns
            WHERE trade_date BETWEEN ? AND ?
            GROUP BY trade_date
            ORDER BY trade_date
            """,
            [
                release.parquet_pattern("daily"),
                release.parquet_pattern("daily"),
                start_date,
                end_date,
                release.parquet_pattern("stock_basic"),
                release.parquet_pattern("daily"),
                release.parquet_pattern("adj_factor"),
                end_date,
                start_date,
                end_date,
            ],
        ).fetchall()
    finally:
        connection.close()
    points = tuple(EqualWeightProxyPoint(row[0], float(row[1]), int(row[2])) for row in rows)
    if not points or points[0].trade_date != start_date or points[-1].trade_date != end_date:
        raise ValueError("equal-weight proxy does not cover the requested trading period")
    stable = {
        "benchmark_id": "PIT_ALL_A_SHARE_DAILY_EQUAL_PROXY",
        "official_wind_index": False,
        "method": (
            "daily equal weight of PIT-listed XSHG/XSHE equities using adjusted close; "
            "suspensions carry forward last adjusted close"
        ),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "observations": len(points),
        "minimum_constituents": min(item.constituent_count for item in points),
        "maximum_constituents": max(item.constituent_count for item in points),
        "points_hash": hashlib.sha256(
            json.dumps([asdict(item) for item in points], default=str, sort_keys=True).encode()
        ).hexdigest(),
    }
    return points, stable
