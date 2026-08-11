from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo

import duckdb

from aquant.backtest import MarketSession
from aquant.data.daily_update import parse_tushare_symbol
from aquant.data.history.tushare import (
    HistoryDatasetManifest,
    HistoryReleaseManifest,
)
from aquant.domain.corporate_actions import CorporateAction, CorporateActionKind
from aquant.domain.market_data import DailyBar, LimitStatus, SecurityStatus
from aquant.strategies.microcap.models import MicrocapObservation, MicrocapSnapshot

_SHANGHAI = ZoneInfo("Asia/Shanghai")


class HistoryReleaseReader:
    """Verified paths and metadata for one immutable historical release."""

    def __init__(self, release_directory: Path) -> None:
        self.directory = release_directory.resolve()
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"history release manifest not found: {manifest_path}")
        self.manifest = HistoryReleaseManifest.from_json(manifest_path.read_text(encoding="utf-8"))
        expected = {dataset for dataset, _checksum, _rows in self.manifest.datasets}
        actual = {
            path.name.removeprefix("dataset=")
            for path in self.directory.glob("dataset=*")
            if path.is_dir()
        }
        if expected != actual:
            raise ValueError("history release dataset directories do not match its manifest")
        self._datasets: dict[str, HistoryDatasetManifest] = {}
        for dataset in expected:
            path = self.directory / f"dataset={dataset}" / "manifest.json"
            parsed = HistoryDatasetManifest.from_json(path.read_text(encoding="utf-8"))
            if parsed.dataset != dataset:
                raise ValueError(f"history dataset manifest identity mismatch: {dataset}")
            self._datasets[dataset] = parsed

    def require(self, *datasets: str) -> None:
        missing = set(datasets) - set(self._datasets)
        if missing:
            raise ValueError(f"history release is missing datasets: {sorted(missing)}")

    def has_dataset(self, dataset: str) -> bool:
        return dataset in self._datasets

    def parquet_pattern(self, dataset: str) -> str:
        try:
            manifest = self._datasets[dataset]
        except KeyError as exc:
            raise ValueError(f"history release has no dataset: {dataset}") from exc
        if len(manifest.files) == 1:
            return str(self.directory / f"dataset={dataset}" / manifest.files[0][0])
        return str(self.directory / f"dataset={dataset}" / "year=*" / "data.parquet")


class DuckDBMicrocapHistory:
    """Read PIT snapshots and short smoke-test sessions from a history release."""

    def __init__(self, release_directory: Path) -> None:
        self.release = HistoryReleaseReader(release_directory)
        self._connection = duckdb.connect(":memory:")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "DuckDBMicrocapHistory":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def trading_dates(self, start_date: date, end_date: date) -> tuple[date, ...]:
        if start_date > end_date:
            raise ValueError("trading date range is inverted")
        self.release.require("trade_cal")
        rows = self._connection.execute(
            """
            SELECT DISTINCT cal_date
            FROM read_parquet(?)
            WHERE is_open
              AND exchange IN ('SSE', 'SZSE')
              AND cal_date BETWEEN ? AND ?
            ORDER BY cal_date
            """,
            [self.release.parquet_pattern("trade_cal"), start_date, end_date],
        ).fetchall()
        return tuple(row[0] for row in rows)

    def snapshot(self, trade_date: date) -> MicrocapSnapshot:
        self.release.require(
            "daily_basic",
            "stock_basic",
            "namechange",
            "suspend_d",
            "fina_indicator",
            "daily",
        )
        rows = self._connection.execute(
            """
            WITH historical_name AS (
                SELECT ts_code, name
                FROM (
                    SELECT
                        ts_code,
                        name,
                        row_number() OVER (
                            PARTITION BY ts_code
                            ORDER BY start_date DESC, coalesce(end_date, DATE '9999-12-31') DESC
                        ) AS row_number
                    FROM read_parquet(?)
                    WHERE start_date <= ?
                      AND (end_date IS NULL OR end_date >= ?)
                )
                WHERE row_number = 1
            ),
            latest_financial AS (
                SELECT ts_code, netprofit_yoy, debt_to_assets
                FROM (
                    SELECT
                        ts_code,
                        netprofit_yoy,
                        debt_to_assets,
                        row_number() OVER (
                            PARTITION BY ts_code
                            ORDER BY end_date DESC, ann_date DESC, coalesce(update_flag, '') DESC
                        ) AS row_number
                    FROM read_parquet(?)
                    WHERE ann_date < ?
                )
                WHERE row_number = 1
            ),
            suspension_events AS (
                SELECT DISTINCT ts_code
                FROM read_parquet(?)
                WHERE trade_date = ?
            ),
            traded AS (
                SELECT DISTINCT ts_code
                FROM read_parquet(?)
                WHERE trade_date = ?
            ),
            suspended AS (
                SELECT suspension_events.ts_code
                FROM suspension_events
                LEFT JOIN traded USING (ts_code)
                WHERE traded.ts_code IS NULL
            ),
            current_basic AS (
                SELECT *
                FROM read_parquet(?)
                WHERE trade_date = ?
            ),
            carried_suspended AS (
                SELECT history.*
                FROM suspended
                JOIN LATERAL (
                    SELECT *
                    FROM read_parquet(?) AS candidate
                    WHERE candidate.ts_code = suspended.ts_code
                      AND candidate.trade_date < ?
                    ORDER BY candidate.trade_date DESC
                    LIMIT 1
                ) AS history ON TRUE
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM current_basic
                    WHERE current_basic.ts_code = suspended.ts_code
                )
            ),
            basic_asof AS (
                SELECT * FROM current_basic
                UNION ALL
                SELECT * FROM carried_suspended
            )
            SELECT
                basic.ts_code,
                instrument.list_date,
                basic.total_market_cap,
                basic.float_market_cap,
                basic.pb,
                financial.netprofit_yoy,
                basic.dividend_yield,
                financial.debt_to_assets,
                basic.turnover_volatility_20d,
                suspended.ts_code IS NOT NULL AS suspended,
                historical_name.name
            FROM basic_asof AS basic
            JOIN read_parquet(?) AS instrument USING (ts_code)
            LEFT JOIN historical_name USING (ts_code)
            LEFT JOIN latest_financial AS financial USING (ts_code)
            LEFT JOIN suspended USING (ts_code)
            WHERE instrument.list_date <= basic.trade_date
            ORDER BY basic.ts_code
            """,
            [
                self.release.parquet_pattern("namechange"),
                trade_date,
                trade_date,
                self.release.parquet_pattern("fina_indicator"),
                trade_date,
                self.release.parquet_pattern("suspend_d"),
                trade_date,
                self.release.parquet_pattern("daily"),
                trade_date,
                self.release.parquet_pattern("daily_basic"),
                trade_date,
                self.release.parquet_pattern("daily_basic"),
                trade_date,
                self.release.parquet_pattern("stock_basic"),
            ],
        ).fetchall()
        if not rows:
            raise ValueError(f"history release has no micro-cap cross-section for {trade_date}")
        available_at = datetime.combine(trade_date, time(15, 0), _SHANGHAI)
        observations = tuple(
            MicrocapObservation(
                symbol=parse_tushare_symbol(row[0]),
                trade_date=trade_date,
                available_at=available_at,
                list_date=row[1],
                total_market_cap=_decimal(row[2]),
                float_market_cap=_optional_decimal(row[3]),
                pb=_optional_decimal(row[4]),
                net_profit_yoy=_optional_decimal(row[5]),
                dividend_yield=_optional_decimal(row[6]),
                debt_to_assets=_optional_decimal(row[7]),
                turnover_volatility_20d=_optional_decimal(row[8]),
                suspended=bool(row[9]),
                is_st=_unsafe_historical_name(row[10]) or _is_st(row[10]),
                is_delisting_risk=(_unsafe_historical_name(row[10]) or _is_delisting_risk(row[10])),
            )
            for row in rows
        )
        return MicrocapSnapshot(trade_date, available_at, observations)

    def snapshots(self, trading_dates: tuple[date, ...]) -> dict[date, MicrocapSnapshot]:
        return {trade_date: self.snapshot(trade_date) for trade_date in trading_dates}

    def sessions(
        self,
        start_date: date,
        end_date: date,
        *,
        ts_codes: tuple[str, ...] | None = None,
    ) -> tuple[MarketSession, ...]:
        """Load sessions, optionally restricted to an explicit immutable symbol universe."""

        self.release.require("daily", "stk_limit", "suspend_d", "namechange")
        codes = tuple(sorted(set(ts_codes or ())))
        if ts_codes is not None and not codes:
            raise ValueError("history session symbol filter must not be empty")
        symbol_filter = " AND {alias}.ts_code = ANY(?)" if codes else ""
        rows = self._connection.execute(
            f"""
            WITH historical_names AS (
                SELECT
                    bar.ts_code,
                    bar.trade_date,
                    name.name,
                    row_number() OVER (
                        PARTITION BY bar.ts_code, bar.trade_date
                        ORDER BY
                            name.start_date DESC,
                            coalesce(name.end_date, DATE '9999-12-31') DESC
                    ) AS row_number
                FROM read_parquet(?) AS bar
                LEFT JOIN read_parquet(?) AS name
                  ON name.ts_code = bar.ts_code
                 AND name.start_date <= bar.trade_date
                 AND (name.end_date IS NULL OR name.end_date >= bar.trade_date)
                WHERE bar.trade_date BETWEEN ? AND ?
                {symbol_filter.format(alias="bar")}
            ),
            limits AS (
                SELECT *
                FROM read_parquet(?) AS limit_row
                WHERE trade_date BETWEEN ? AND ?
                {symbol_filter.format(alias="limit_row")}
            ),
            suspensions AS (
                SELECT DISTINCT ts_code, trade_date
                FROM read_parquet(?) AS suspension_row
                WHERE trade_date BETWEEN ? AND ?
                {symbol_filter.format(alias="suspension_row")}
            )
            SELECT
                bar.ts_code,
                bar.trade_date,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.amount,
                bar.prior_20d_average_volume,
                limits.up_limit,
                limits.down_limit,
                suspensions.ts_code IS NOT NULL AS suspended,
                historical_names.name
            FROM read_parquet(?) AS bar
            LEFT JOIN limits USING (ts_code, trade_date)
            LEFT JOIN suspensions USING (ts_code, trade_date)
            LEFT JOIN historical_names
              ON historical_names.ts_code = bar.ts_code
             AND historical_names.trade_date = bar.trade_date
             AND historical_names.row_number = 1
            WHERE bar.trade_date BETWEEN ? AND ?
            {symbol_filter.format(alias="bar")}
            ORDER BY bar.trade_date, bar.ts_code
            """,
            [
                self.release.parquet_pattern("daily"),
                self.release.parquet_pattern("namechange"),
                start_date,
                end_date,
                *([codes] if codes else []),
                self.release.parquet_pattern("stk_limit"),
                start_date,
                end_date,
                *([codes] if codes else []),
                self.release.parquet_pattern("suspend_d"),
                start_date,
                end_date,
                *([codes] if codes else []),
                self.release.parquet_pattern("daily"),
                start_date,
                end_date,
                *([codes] if codes else []),
            ],
        ).fetchall()
        grouped_bars: dict[date, list[DailyBar]] = {}
        grouped_statuses: dict[date, list[SecurityStatus]] = {}
        for row in rows:
            symbol = parse_tushare_symbol(row[0])
            trade_date = row[1]
            opening = _decimal(row[2])
            up_limit = _optional_decimal(row[9])
            down_limit = _optional_decimal(row[10])
            grouped_bars.setdefault(trade_date, []).append(
                DailyBar(
                    symbol=symbol,
                    trade_date=trade_date,
                    open=opening,
                    high=_decimal(row[3]),
                    low=_decimal(row[4]),
                    close=_decimal(row[5]),
                    volume=_decimal(row[6]),
                    amount=_decimal(row[7]),
                )
            )
            grouped_statuses.setdefault(trade_date, []).append(
                SecurityStatus(
                    symbol=symbol,
                    trade_date=trade_date,
                    suspended=False,
                    is_st=_unsafe_historical_name(row[12]) or _is_st(row[12]),
                    limit_status=_limit_status(opening, up_limit, down_limit),
                    prior_20d_average_volume=_optional_decimal(row[8]),
                )
            )
        dates = sorted(grouped_bars)
        if not dates:
            raise ValueError("history release has no bars in the requested smoke window")
        grouped_actions = self.corporate_actions(start_date, end_date, ts_codes=codes or None)
        return tuple(
            MarketSession(
                trade_date=value,
                open_at=datetime.combine(value, time(9, 30), _SHANGHAI),
                close_at=datetime.combine(value, time(15, 0), _SHANGHAI),
                bars=tuple(grouped_bars[value]),
                statuses=tuple(grouped_statuses[value]),
                corporate_actions=grouped_actions.get(value, ()),
            )
            for value in dates
        )

    def corporate_actions(
        self,
        start_date: date,
        end_date: date,
        *,
        ts_codes: tuple[str, ...] | None = None,
    ) -> dict[date, tuple[CorporateAction, ...]]:
        if not self.release.has_dataset("dividend"):
            return {}
        codes = tuple(sorted(set(ts_codes or ())))
        if ts_codes is not None and not codes:
            raise ValueError("corporate-action symbol filter must not be empty")
        rows = self._connection.execute(
            f"""
            SELECT DISTINCT
                dividend.ts_code,
                dividend.stk_div,
                dividend.cash_div_tax,
                dividend.ex_date,
                dividend.pay_date,
                coalesce(bar.open, prior.close) AS cash_in_lieu_reference
            FROM read_parquet(?) AS dividend
            LEFT JOIN read_parquet(?) AS bar
              ON bar.ts_code = dividend.ts_code
             AND bar.trade_date = dividend.ex_date
            LEFT JOIN LATERAL (
                SELECT prior_bar.close
                FROM read_parquet(?) AS prior_bar
                WHERE prior_bar.ts_code = dividend.ts_code
                  AND prior_bar.trade_date < dividend.ex_date
                ORDER BY prior_bar.trade_date DESC
                LIMIT 1
            ) AS prior ON true
            WHERE div_proc = '实施'
              AND (
                    (dividend.ex_date BETWEEN ? AND ?)
                 OR (dividend.pay_date BETWEEN ? AND ?)
              )
              {"AND dividend.ts_code = ANY(?)" if codes else ""}
            ORDER BY coalesce(dividend.ex_date, dividend.pay_date), dividend.ts_code
            """,
            [
                self.release.parquet_pattern("dividend"),
                self.release.parquet_pattern("daily"),
                self.release.parquet_pattern("daily"),
                start_date,
                end_date,
                start_date,
                end_date,
                *([codes] if codes else []),
            ],
        ).fetchall()
        grouped: dict[date, list[CorporateAction]] = {}
        for ts_code, stock_ratio, cash_per_share, ex_date, pay_date, cash_reference in rows:
            symbol = parse_tushare_symbol(ts_code)
            identity = f"{ts_code}:{ex_date}:{pay_date}:{stock_ratio}:{cash_per_share}"
            entitlement_id = uuid5(NAMESPACE_URL, f"aquant:dividend:{identity}:cash")
            if ex_date is not None and start_date <= ex_date <= end_date:
                occurred_at = datetime.combine(ex_date, time(9, 30), _SHANGHAI)
                cash = _optional_decimal(cash_per_share)
                stock = _optional_decimal(stock_ratio)
                if cash is not None and cash > 0:
                    grouped.setdefault(ex_date, []).append(
                        CorporateAction(
                            entitlement_id,
                            symbol,
                            CorporateActionKind.CASH_DIVIDEND_ENTITLEMENT,
                            occurred_at,
                            cash_per_share=cash,
                        )
                    )
                if stock is not None and stock > 0:
                    grouped.setdefault(ex_date, []).append(
                        CorporateAction(
                            uuid5(NAMESPACE_URL, f"aquant:dividend:{identity}:stock"),
                            symbol,
                            CorporateActionKind.STOCK_DIVIDEND,
                            occurred_at,
                            ratio=stock,
                            cash_in_lieu_price=_optional_decimal(cash_reference),
                        )
                    )
            if (
                pay_date is not None
                and cash_per_share is not None
                and Decimal(cash_per_share) > 0
                and start_date <= pay_date <= end_date
            ):
                grouped.setdefault(pay_date, []).append(
                    CorporateAction(
                        uuid5(NAMESPACE_URL, f"aquant:dividend:{identity}:payment"),
                        symbol,
                        CorporateActionKind.CASH_DIVIDEND_PAYMENT,
                        datetime.combine(pay_date, time(9, 30), _SHANGHAI),
                        reference_action_id=entitlement_id,
                    )
                )
        return {
            trade_date: tuple(
                sorted(actions, key=lambda item: (item.occurred_at, str(item.action_id)))
            )
            for trade_date, actions in grouped.items()
        }


def _decimal(value: object) -> Decimal:
    resolved = Decimal(str(value))
    if not resolved.is_finite():
        raise ValueError("history numeric field must be finite")
    return resolved


def _optional_decimal(value: object) -> Decimal | None:
    return None if value is None else _decimal(value)


def _unsafe_historical_name(value: object) -> bool:
    return not isinstance(value, str) or not value.strip()


def _is_st(value: object) -> bool:
    return isinstance(value, str) and "ST" in value.upper()


def _is_delisting_risk(value: object) -> bool:
    return isinstance(value, str) and ("*ST" in value.upper() or "退" in value)


def _limit_status(
    opening: Decimal,
    up_limit: Decimal | None,
    down_limit: Decimal | None,
) -> LimitStatus:
    if up_limit is None or down_limit is None:
        return LimitStatus.UNKNOWN
    if opening >= up_limit:
        return LimitStatus.LIMIT_UP
    if opening <= down_limit:
        return LimitStatus.LIMIT_DOWN
    return LimitStatus.NONE
