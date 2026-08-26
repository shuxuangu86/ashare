import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class IndustryQualityThresholds:
    minimum_market_row_coverage: float = 0.95
    minimum_year_market_row_coverage: float = 0.95
    maximum_conflicting_market_rows: int = 0
    maximum_invalid_intervals: int = 0
    maximum_orphan_industry_codes: int = 0
    market_exchanges: tuple[str, ...] = ("XSHG", "XSHE")


def validate_industry_release(
    *,
    industry_release: Path,
    history_release: Path,
    start_date: date,
    end_date: date,
    thresholds: IndustryQualityThresholds,
) -> dict[str, Any]:
    manifest = json.loads((industry_release / "manifest.json").read_text(encoding="utf-8"))
    memberships = industry_release / "industry_memberships_pit.parquet"
    classifications = industry_release / "industry_classifications.parquet"
    daily_glob = str(history_release / "dataset=daily" / "**" / "*.parquet")
    connection = duckdb.connect()
    try:
        coverage = connection.execute(
            """
            WITH market AS (
                SELECT ts_code, trade_date
                FROM read_parquet(?, hive_partitioning=false)
                WHERE trade_date BETWEEN ? AND ?
                  AND exchange IN (SELECT unnest(?))
            ), joined AS (
                SELECT d.ts_code, d.trade_date, count(m.industry_code) AS matches
                FROM market d
                LEFT JOIN read_parquet(?, hive_partitioning=false) m
                  ON m.ts_code = d.ts_code
                 AND m.industry_level = 'L1'
                 AND m.effective_from <= d.trade_date
                 AND (m.effective_to IS NULL OR d.trade_date < m.effective_to)
                 AND m.available_at <= d.trade_date + INTERVAL 15 HOUR
                GROUP BY d.ts_code, d.trade_date
            )
            SELECT
                count(*) AS market_rows,
                count(*) FILTER (WHERE matches = 1) AS covered_rows,
                count(*) FILTER (WHERE matches = 0) AS missing_rows,
                count(*) FILTER (WHERE matches > 1) AS conflicting_rows,
                count(DISTINCT trade_date) AS trading_days,
                count(DISTINCT ts_code) AS securities
            FROM joined
            """,
            [
                daily_glob,
                start_date,
                end_date,
                list(thresholds.market_exchanges),
                str(memberships),
            ],
        ).fetchone()
        assert coverage is not None
        yearly_rows = connection.execute(
            """
            WITH market AS (
                SELECT ts_code, trade_date
                FROM read_parquet(?, hive_partitioning=false)
                WHERE trade_date BETWEEN ? AND ?
                  AND exchange IN (SELECT unnest(?))
            ), joined AS (
                SELECT d.ts_code, d.trade_date, count(m.industry_code) AS matches
                FROM market d
                LEFT JOIN read_parquet(?, hive_partitioning=false) m
                  ON m.ts_code = d.ts_code
                 AND m.industry_level = 'L1'
                 AND m.effective_from <= d.trade_date
                 AND (m.effective_to IS NULL OR d.trade_date < m.effective_to)
                 AND m.available_at <= d.trade_date + INTERVAL 15 HOUR
                GROUP BY d.ts_code, d.trade_date
            )
            SELECT year(trade_date), count(*),
                   count(*) FILTER (WHERE matches = 0),
                   count(*) FILTER (WHERE matches > 1)
            FROM joined GROUP BY 1 ORDER BY 1
            """,
            [
                daily_glob,
                start_date,
                end_date,
                list(thresholds.market_exchanges),
                str(memberships),
            ],
        ).fetchall()
    finally:
        connection.close()
    membership_rows = pq.ParquetFile(memberships).read().to_pylist()
    classification_rows = pq.ParquetFile(classifications).read().to_pylist()
    invalid_intervals = sum(
        row["effective_to"] is not None and row["effective_from"] >= row["effective_to"]
        for row in membership_rows
    )
    available_after_effective_session = sum(
        row["available_at"].date() > row["effective_from"] for row in membership_rows
    )
    known_codes = {row["industry_code"] for row in classification_rows}
    orphan_codes = sorted(
        {row["industry_code"] for row in membership_rows if row["industry_code"] not in known_codes}
    )
    switch_counts: dict[str, int] = {}
    for row in membership_rows:
        if row["industry_level"] == "L1":
            switch_counts[row["ts_code"]] = switch_counts.get(row["ts_code"], -1) + 1
    market_rows, covered_rows, missing_rows, conflicts, trading_days, securities = map(
        int, coverage
    )
    ratio = covered_rows / market_rows if market_rows else 0.0
    failures: list[str] = []
    if ratio < thresholds.minimum_market_row_coverage:
        failures.append("MARKET_ROW_COVERAGE_BELOW_THRESHOLD")
    yearly_coverage = {
        int(year): 1.0 - int(missing) / int(rows) if rows else 0.0
        for year, rows, missing, _ in yearly_rows
    }
    if any(
        coverage < thresholds.minimum_year_market_row_coverage
        for coverage in yearly_coverage.values()
    ):
        failures.append("YEAR_MARKET_ROW_COVERAGE_BELOW_THRESHOLD")
    if conflicts > thresholds.maximum_conflicting_market_rows:
        failures.append("CONFLICTING_MARKET_MEMBERSHIP")
    if invalid_intervals > thresholds.maximum_invalid_intervals:
        failures.append("INVALID_EFFECTIVE_INTERVAL")
    if len(orphan_codes) > thresholds.maximum_orphan_industry_codes:
        failures.append("ORPHAN_INDUSTRY_CODE")
    stable = {
        "schema_version": "aquant.industry-pit-quality.v1",
        "status": "PASS" if not failures else "BLOCKED",
        "data_release_id": manifest["data_release_id"],
        "industry_release_id": manifest["release_id"],
        "industry_release_hash": manifest["content_hash"],
        "config": asdict(thresholds),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "market_rows": market_rows,
        "covered_rows": covered_rows,
        "missing_rows": missing_rows,
        "conflicting_rows": conflicts,
        "market_row_coverage": ratio,
        "trading_days": trading_days,
        "securities": securities,
        "years": {
            str(year): {
                "market_rows": int(rows),
                "missing_rows": int(missing),
                "conflicting_rows": int(year_conflicts),
                "missing_rate": int(missing) / int(rows) if rows else 0.0,
                "coverage": yearly_coverage[int(year)],
            }
            for year, rows, missing, year_conflicts in yearly_rows
        },
        "invalid_intervals": invalid_intervals,
        "available_after_effective_session": available_after_effective_session,
        "orphan_industry_codes": orphan_codes,
        "industry_switches": {
            "securities": len(switch_counts),
            "maximum": max(switch_counts.values(), default=0),
            "mean": (sum(switch_counts.values()) / len(switch_counts) if switch_counts else 0.0),
        },
        "failure_reason_codes": failures,
    }
    return {
        **stable,
        "created_at": datetime.now(UTC).isoformat(timespec="microseconds"),
        "content_hash": _hash(stable),
    }


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
