from bisect import bisect_left
from datetime import date, datetime
from pathlib import Path

import duckdb
import numpy as np
import numpy.typing as npt

from aquant.data.history.microcap import HistoryReleaseReader
from aquant.domain.data_release import DataReleaseId
from aquant.factors.atomic.models import FactorPanelInput

_FIELD_SOURCE = {
    "open": "daily",
    "high": "daily",
    "low": "daily",
    "close": "daily",
    "volume": "daily",
    "amount": "daily",
    "total_market_cap": "basic",
    "float_market_cap": "basic",
    "pb": "basic",
    "turnover_rate": "basic",
    "dividend_yield": "basic",
    "netprofit_yoy": "financial",
    "debt_to_assets": "financial",
    "up_limit": "limits",
    "down_limit": "limits",
}


class StandardPITFactorLoader:
    """Load aligned factor panels from one immutable Standard/PIT release."""

    def __init__(self, release_directory: Path) -> None:
        self.release = HistoryReleaseReader(release_directory)

    def load(
        self,
        *,
        fields: tuple[str, ...],
        start_date: date,
        end_date: date,
        as_of_time: datetime,
        universe_id: str,
        data_release_id: DataReleaseId,
    ) -> FactorPanelInput:
        del data_release_id
        if start_date > end_date or end_date > as_of_time.date():
            raise ValueError("factor load range must be ordered and visible as-of")
        if universe_id != "all_a_share":
            raise ValueError("first release supports only all_a_share")
        unknown = set(fields) - _FIELD_SOURCE.keys()
        if unknown:
            raise ValueError(f"unsupported Standard/PIT fields: {sorted(unknown)}")
        required = {"daily"} | {
            {"basic": "daily_basic", "financial": "fina_indicator", "limits": "stk_limit"}[source]
            for source in {_FIELD_SOURCE[field] for field in fields}
            if source != "daily"
        }
        self.release.require(*sorted(required))
        sources = {_FIELD_SOURCE[field] for field in fields}
        query_fields = tuple(field for field in fields if _FIELD_SOURCE[field] != "financial")
        selections = ", ".join(
            f"{_FIELD_SOURCE[field]}.{field} AS {field}" for field in query_fields
        )
        joins: list[str] = []
        parameters: list[object] = [
            self.release.parquet_pattern("daily"),
            start_date,
            end_date,
        ]
        if "basic" in sources:
            joins.append(
                """
                LEFT JOIN read_parquet(?) AS basic
                  ON basic.trade_date = daily.trade_date
                 AND basic.ts_code = daily.ts_code
                 AND basic.trade_date BETWEEN ? AND ?
                """
            )
            parameters.extend([self.release.parquet_pattern("daily_basic"), start_date, end_date])
        if "limits" in sources:
            joins.append(
                """
                LEFT JOIN read_parquet(?) AS limits
                  ON limits.trade_date = daily.trade_date
                 AND limits.ts_code = daily.ts_code
                 AND limits.trade_date BETWEEN ? AND ?
                """
            )
            parameters.extend([self.release.parquet_pattern("stk_limit"), start_date, end_date])
        selection_clause = f", {selections}" if selections else ""
        query = f"""
            SELECT daily.trade_date, daily.ts_code {selection_clause}
            FROM read_parquet(?) AS daily
            {" ".join(joins)}
            WHERE daily.trade_date BETWEEN ? AND ?
              AND daily.exchange IN ('XSHG', 'XSHE')
            ORDER BY daily.trade_date, daily.ts_code
        """
        connection = duckdb.connect(":memory:")
        try:
            daily_pattern = self.release.parquet_pattern("daily")
            trade_dates = tuple(
                row[0]
                for row in connection.execute(
                    """
                    SELECT DISTINCT trade_date
                    FROM read_parquet(?)
                    WHERE trade_date BETWEEN ? AND ?
                      AND exchange IN ('XSHG', 'XSHE')
                    ORDER BY trade_date
                    """,
                    [daily_pattern, start_date, end_date],
                ).fetchall()
            )
            ts_codes = tuple(
                row[0]
                for row in connection.execute(
                    """
                    SELECT DISTINCT ts_code
                    FROM read_parquet(?)
                    WHERE trade_date BETWEEN ? AND ?
                      AND exchange IN ('XSHG', 'XSHE')
                    ORDER BY ts_code
                    """,
                    [daily_pattern, start_date, end_date],
                ).fetchall()
            )
            if not trade_dates or not ts_codes:
                raise ValueError("Standard/PIT release has no rows in requested range")
            date_index = {value: index for index, value in enumerate(trade_dates)}
            code_index = {value: index for index, value in enumerate(ts_codes)}
            arrays: dict[str, npt.NDArray[np.float64]] = {
                field: np.full(
                    (len(trade_dates), len(ts_codes)),
                    np.nan,
                    dtype=np.float64,
                )
                for field in fields
            }
            batches = connection.execute(
                query,
                [
                    parameters[0],
                    *parameters[3:],
                    parameters[1],
                    parameters[2],
                ],
            ).fetch_record_batch(rows_per_batch=100_000)
            for batch in batches:
                batch_dates = batch.column(0).to_pylist()
                batch_codes = batch.column(1).to_pylist()
                date_positions = np.fromiter(
                    (date_index[value] for value in batch_dates),
                    dtype=np.int64,
                )
                code_positions = np.fromiter(
                    (code_index[value] for value in batch_codes),
                    dtype=np.int64,
                )
                for offset, field in enumerate(query_fields, start=2):
                    values = np.asarray(batch.column(offset).to_pylist(), dtype=np.float64)
                    arrays[field][date_positions, code_positions] = values
            if "financial" in sources:
                self._fill_financial(
                    connection,
                    fields=fields,
                    trade_dates=trade_dates,
                    ts_codes=ts_codes,
                    arrays=arrays,
                    end_date=end_date,
                )
        finally:
            connection.close()
        return FactorPanelInput(trade_dates, ts_codes, arrays)

    def _fill_financial(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        fields: tuple[str, ...],
        trade_dates: tuple[date, ...],
        ts_codes: tuple[str, ...],
        arrays: dict[str, npt.NDArray[np.float64]],
        end_date: date,
    ) -> None:
        financial_fields = tuple(field for field in fields if _FIELD_SOURCE[field] == "financial")
        selections = ", ".join(financial_fields)
        rows = connection.execute(
            f"""
            SELECT
                ts_code,
                ann_date,
                end_date,
                coalesce(update_flag, ''),
                {selections}
            FROM read_parquet(?)
            WHERE ann_date < ?
            ORDER BY ts_code, ann_date, end_date, coalesce(update_flag, '')
            """,
            [self.release.parquet_pattern("fina_indicator"), end_date],
        ).fetchall()
        events: dict[
            str,
            list[tuple[date, tuple[date, date, str], tuple[float, ...]]],
        ] = {}
        for row in rows:
            key: tuple[date, date, str] = (row[2], row[1], row[3])
            values = tuple(float(value) if value is not None else np.nan for value in row[4:])
            events.setdefault(row[0], []).append((row[1], key, values))
        for code_position, ts_code in enumerate(ts_codes):
            candidates = events.get(ts_code)
            if not candidates:
                continue
            announcement_dates: list[date] = []
            cumulative_values: list[tuple[float, ...]] = []
            best_key: tuple[date, date, str] | None = None
            best_values: tuple[float, ...] = ()
            for announced, key, values in candidates:
                if best_key is None or key > best_key:
                    best_key, best_values = key, values
                announcement_dates.append(announced)
                cumulative_values.append(best_values)
            for date_position, trade_date in enumerate(trade_dates):
                event_position = bisect_left(announcement_dates, trade_date) - 1
                if event_position < 0:
                    continue
                for field_position, field in enumerate(financial_fields):
                    arrays[field][date_position, code_position] = cumulative_values[event_position][
                        field_position
                    ]
