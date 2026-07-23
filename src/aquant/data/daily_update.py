import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any

from aquant.data.ingestion import ArchivedPage, TushareBulkArchiver
from aquant.data.quality import DailyBarQualityGate, QualityIssue, QualityReport, Severity
from aquant.domain.data_release import DataReleaseId
from aquant.domain.enums import Exchange
from aquant.domain.identifiers import Symbol
from aquant.domain.market_data import DailyBar

API_PAGE_SIZES: Mapping[str, int] = {
    "stock_basic": 6000,
    "namechange": 5000,
    "trade_cal": 6000,
    "daily": 6000,
    "adj_factor": 6000,
    "daily_basic": 6000,
    "suspend_d": 5000,
    "stk_limit": 5800,
    "limit_list_d": 5000,
    "moneyflow": 6000,
    "income": 5000,
    "balancesheet": 5000,
    "cashflow": 5000,
    "fina_indicator": 5000,
    "forecast": 2000,
    "express": 1000,
    "disclosure_date": 3000,
    "dividend": 2000,
    "share_float": 6000,
    "index_daily": 6000,
    "index_dailybasic": 3000,
}

MARKET_APIS: tuple[tuple[str, bool, bool], ...] = (
    ("daily", True, True),
    ("adj_factor", True, True),
    ("daily_basic", True, True),
    ("suspend_d", True, False),
    ("stk_limit", True, True),
    ("limit_list_d", False, False),
    ("moneyflow", False, False),
)
ANNOUNCEMENT_APIS: tuple[tuple[str, str, bool], ...] = (
    ("income", "ann_date", True),
    ("balancesheet", "ann_date", True),
    ("cashflow", "ann_date", True),
    ("fina_indicator", "ann_date", True),
    ("forecast", "ann_date", False),
    ("express", "ann_date", False),
    ("disclosure_date", "actual_date", False),
    ("dividend", "ann_date", False),
    ("share_float", "ann_date", False),
)
BENCHMARKS: tuple[str, ...] = (
    "000001.SH",
    "399001.SZ",
    "399006.SZ",
    "000016.SH",
    "000300.SH",
    "000905.SH",
    "000852.SH",
)


class DailyUpdateMode(StrEnum):
    CLOSE = "close"
    MORNING = "morning"


@dataclass(frozen=True, slots=True)
class UpdateJob:
    api_name: str
    params: Mapping[str, Any]
    group: str
    required: bool
    expect_rows: bool = False

    def __post_init__(self) -> None:
        if not self.api_name.strip() or not self.group.strip():
            raise ValueError("daily update job identity must not be blank")
        object.__setattr__(self, "params", dict(self.params))


@dataclass(frozen=True, slots=True)
class UpdateJobResult:
    job: UpdateJob
    pages: tuple[ArchivedPage, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if self.error is not None and not self.error.strip():
            raise ValueError("job error must be non-blank when present")
        if self.error is not None and self.pages:
            raise ValueError("failed job must not expose completed pages")

    @property
    def row_count(self) -> int:
        return sum(page.row_count for page in self.pages)

    @property
    def resumed_pages(self) -> int:
        return sum(page.resumed for page in self.pages)


@dataclass(frozen=True, slots=True)
class DailyUpdateValidation:
    report: QualityReport
    bars: tuple[DailyBar, ...]


def select_market_dates(
    calendar_rows: Iterable[Mapping[str, Any]],
    *,
    as_of_date: date,
    mode: DailyUpdateMode,
    morning_lookback_sessions: int = 3,
) -> tuple[date, ...]:
    if morning_lookback_sessions <= 0:
        raise ValueError("morning lookback sessions must be positive")
    open_dates = sorted(
        {
            _yyyymmdd(row.get("cal_date"), field="cal_date")
            for row in calendar_rows
            if str(row.get("is_open")) == "1"
        }
    )
    if mode is DailyUpdateMode.CLOSE:
        return (as_of_date,) if as_of_date in open_dates else ()
    eligible = tuple(value for value in open_dates if value < as_of_date)
    return eligible[-morning_lookback_sessions:]


def announcement_dates(
    *,
    as_of_date: date,
    market_dates: Sequence[date],
    mode: DailyUpdateMode,
) -> tuple[date, ...]:
    if mode is DailyUpdateMode.CLOSE:
        return (as_of_date,)
    return tuple(market_dates)


def build_update_plan(
    *,
    market_dates: Sequence[date],
    announcement_check_dates: Sequence[date],
) -> tuple[UpdateJob, ...]:
    jobs: list[UpdateJob] = [
        UpdateJob(
            "stock_basic",
            {"list_status": status},
            "reference",
            required=False,
            expect_rows=True,
        )
        for status in ("L", "D", "P")
    ]
    jobs.append(UpdateJob("namechange", {}, "reference", required=False))

    for session in market_dates:
        trade_date = session.strftime("%Y%m%d")
        jobs.extend(
            UpdateJob(
                api_name,
                {"trade_date": trade_date},
                "market",
                required=required,
                expect_rows=expect_rows,
            )
            for api_name, required, expect_rows in MARKET_APIS
        )
        for benchmark in BENCHMARKS:
            jobs.append(
                UpdateJob(
                    "index_daily",
                    {
                        "ts_code": benchmark,
                        "start_date": trade_date,
                        "end_date": trade_date,
                    },
                    "index",
                    required=True,
                    expect_rows=True,
                )
            )
            jobs.append(
                UpdateJob(
                    "index_dailybasic",
                    {
                        "ts_code": benchmark,
                        "start_date": trade_date,
                        "end_date": trade_date,
                    },
                    "index",
                    required=False,
                )
            )

    for check_date in announcement_check_dates:
        value = check_date.strftime("%Y%m%d")
        jobs.extend(
            UpdateJob(
                api_name,
                {date_parameter: value},
                "announcements",
                required=required,
            )
            for api_name, date_parameter, required in ANNOUNCEMENT_APIS
        )
    return tuple(jobs)


def archive_job(archiver: TushareBulkArchiver, job: UpdateJob) -> UpdateJobResult:
    try:
        pages = tuple(
            archiver.archive_paginated(
                job.api_name,
                job.params,
                page_size=API_PAGE_SIZES.get(job.api_name, 1000),
            )
        )
    except Exception as exc:
        return UpdateJobResult(job, error=f"{type(exc).__name__}: {exc}")
    return UpdateJobResult(job, pages)


def validate_update_results(
    archiver: TushareBulkArchiver,
    results: Iterable[UpdateJobResult],
    *,
    market_dates: Sequence[date],
) -> DailyUpdateValidation:
    resolved = tuple(results)
    issues: list[QualityIssue] = []
    checked_records = 0
    bars: list[DailyBar] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for result in resolved:
        job_key = _job_label(result.job)
        if result.error is not None:
            issues.append(
                QualityIssue(
                    Severity.ERROR if result.job.required else Severity.WARNING,
                    "REQUIRED_API_FAILED" if result.job.required else "OPTIONAL_API_FAILED",
                    job_key,
                    result.error,
                )
            )
            continue
        checked_records += result.row_count
        if result.job.expect_rows and result.row_count == 0:
            issues.append(
                QualityIssue(
                    Severity.ERROR if result.job.required else Severity.WARNING,
                    "REQUIRED_API_EMPTY" if result.job.required else "OPTIONAL_API_EMPTY",
                    job_key,
                    "API returned no rows for a dataset expected to be non-empty",
                )
            )
        for page in result.pages:
            try:
                records = archiver.read_items(page)
            except Exception as exc:
                issues.append(
                    QualityIssue(
                        Severity.ERROR,
                        "RAW_PAYLOAD_UNREADABLE",
                        job_key,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            for record in records:
                _validate_trade_date(result.job, record, issues)
                record_key = _record_key(result.job.api_name, record)
                if record_key is not None:
                    if record_key in seen_keys:
                        issues.append(
                            QualityIssue(
                                Severity.ERROR,
                                "DUPLICATE_API_PRIMARY_KEY",
                                "|".join(record_key),
                                "duplicate record across daily update pages",
                            )
                        )
                    seen_keys.add(record_key)
                if result.job.api_name == "daily":
                    try:
                        bars.append(parse_daily_bar(record))
                    except (InvalidOperation, TypeError, ValueError) as exc:
                        issues.append(
                            QualityIssue(
                                Severity.ERROR,
                                "DAILY_BAR_INVALID",
                                str(record.get("ts_code") or job_key),
                                str(exc),
                            )
                        )

    bar_report = DailyBarQualityGate().validate(bars, open_dates=market_dates)
    issues.extend(bar_report.issues)
    return DailyUpdateValidation(
        QualityReport("tushare_daily_update", checked_records, tuple(issues)),
        tuple(bars),
    )


def parse_daily_bar(record: Mapping[str, Any]) -> DailyBar:
    return DailyBar(
        symbol=parse_tushare_symbol(_required_text(record, "ts_code")),
        trade_date=_yyyymmdd(record.get("trade_date"), field="trade_date"),
        open=Decimal(str(record.get("open"))),
        high=Decimal(str(record.get("high"))),
        low=Decimal(str(record.get("low"))),
        close=Decimal(str(record.get("close"))),
        volume=Decimal(str(record.get("vol"))),
        amount=Decimal(str(record.get("amount"))),
    )


def parse_tushare_symbol(value: str) -> Symbol:
    try:
        code, suffix = value.strip().upper().split(".", maxsplit=1)
        exchange = {
            "SH": Exchange.XSHG,
            "SZ": Exchange.XSHE,
            "BJ": Exchange.XBSE,
        }[suffix]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"unsupported Tushare symbol: {value!r}") from exc
    return Symbol(code, exchange)


def next_release_id(root: Path, release_date: date) -> DataReleaseId:
    prefix = f"cn_equity_{release_date.strftime('%Y%m%d')}_"
    sequences: list[int] = []
    if root.is_dir():
        for candidate in root.iterdir():
            if candidate.is_dir() and candidate.name.startswith(prefix):
                suffix = candidate.name.removeprefix(prefix)
                if len(suffix) == 3 and suffix.isdigit():
                    sequences.append(int(suffix))
    next_sequence = max(sequences, default=0) + 1
    if next_sequence > 999:
        raise RuntimeError(f"daily release sequence exhausted for {release_date.isoformat()}")
    return DataReleaseId(f"{prefix}{next_sequence:03d}")


def raw_batch_ids(pages: Iterable[ArchivedPage]) -> tuple[str, ...]:
    values: set[str] = set()
    for page in pages:
        payload = json.loads(page.manifest_path.read_text(encoding="utf-8"))
        batch_id = payload.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError(f"raw manifest has no batch_id: {page.manifest_path}")
        values.add(batch_id)
    return tuple(sorted(values))


def result_to_dict(result: UpdateJobResult) -> dict[str, Any]:
    return {
        "api": result.job.api_name,
        "group": result.job.group,
        "params": dict(result.job.params),
        "required": result.job.required,
        "status": "FAILED" if result.error else "COMPLETED",
        "error": result.error,
        "pages": len(result.pages),
        "rows": result.row_count,
        "resumed_pages": result.resumed_pages,
        "raw_batches": [
            {
                "payload": str(page.payload_path),
                "manifest": str(page.manifest_path),
                "rows": page.row_count,
            }
            for page in result.pages
        ],
    }


def _validate_trade_date(
    job: UpdateJob,
    record: Mapping[str, Any],
    issues: list[QualityIssue],
) -> None:
    requested = job.params.get("trade_date")
    actual = record.get("trade_date")
    if requested is not None and actual is not None and str(actual) != str(requested):
        issues.append(
            QualityIssue(
                Severity.ERROR,
                "UNEXPECTED_TRADE_DATE",
                _job_label(job),
                f"requested {requested}, received {actual}",
            )
        )


def _record_key(api_name: str, record: Mapping[str, Any]) -> tuple[str, str, str] | None:
    ts_code = record.get("ts_code")
    trade_date = record.get("trade_date")
    if ts_code is None or trade_date is None:
        return None
    return api_name, str(ts_code), str(trade_date)


def _job_label(job: UpdateJob) -> str:
    parameters = ",".join(f"{key}={value}" for key, value in sorted(job.params.items()))
    return f"{job.api_name}[{parameters}]"


def _required_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing required text field: {field}")
    return value.strip()


def _yyyymmdd(value: object, *, field: str) -> date:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise ValueError(f"invalid {field}: {value!r}")
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
