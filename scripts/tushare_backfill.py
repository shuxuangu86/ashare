import argparse
import fcntl
import json
import os
import shutil
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from aquant.data.ingestion import ArchivedPage, TushareApiError, TushareBulkArchiver

REFERENCE_JOBS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("stock_basic", {"list_status": "L"}),
    ("stock_basic", {"list_status": "D"}),
    ("stock_basic", {"list_status": "P"}),
    ("namechange", {}),
    ("stock_company", {"exchange": "SSE"}),
    ("stock_company", {"exchange": "SZSE"}),
    ("stock_company", {"exchange": "BSE"}),
)

API_PAGE_SIZES = {
    "stock_basic": 6000,
    "trade_cal": 6000,
    "namechange": 5000,
    "stock_company": 4500,
    "index_basic": 5000,
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
    "index_weight": 5000,
    "index_classify": 5000,
    "index_member": 5000,
}

MARKET_APIS = (
    "daily",
    "adj_factor",
    "daily_basic",
    "suspend_d",
    "stk_limit",
    "limit_list_d",
    "moneyflow",
)
FINANCIAL_APIS = (
    "income",
    "balancesheet",
    "cashflow",
    "fina_indicator",
    "forecast",
    "express",
    "disclosure_date",
)
INDEX_MARKETS = ("SSE", "SZSE", "CSI", "CICC", "SW", "MSCI", "OTH")
BENCHMARKS = (
    "000001.SH",
    "399001.SZ",
    "399006.SZ",
    "000016.SH",
    "000300.SH",
    "000905.SH",
    "000852.SH",
)
WEIGHTED_BENCHMARKS = ("000016.SH", "000300.SH", "000905.SH", "000852.SH")


def _log(event: str, **payload: object) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                **payload,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _progress_timing(*, started_at: float, completed: int, total: int) -> dict[str, object]:
    now = datetime.now().astimezone()
    elapsed_seconds = max(time.monotonic() - started_at, 0.001)
    sessions_per_minute = completed / elapsed_seconds * 60
    remaining_seconds = (total - completed) / sessions_per_minute * 60 if sessions_per_minute else 0
    return {
        "checkpoint_finished_at": now.isoformat(timespec="seconds"),
        "elapsed_minutes": round(elapsed_seconds / 60, 2),
        "sessions_per_minute": round(sessions_per_minute, 2),
        "estimated_api_finish_at": (now + timedelta(seconds=remaining_seconds)).isoformat(
            timespec="seconds"
        ),
    }


def _months(start_date: str, end_date: str) -> tuple[tuple[str, str], ...]:
    start_year, start_month = int(start_date[:4]), int(start_date[4:6])
    end_year, end_month = int(end_date[:4]), int(end_date[4:6])
    values = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        if month == 12:
            next_year, next_month = year + 1, 1
        else:
            next_year, next_month = year, month + 1
        last_day = (date(next_year, next_month, 1) - timedelta(days=1)).day
        values.append((f"{year:04d}{month:02d}01", f"{year:04d}{month:02d}{last_day:02d}"))
        year, month = next_year, next_month
    return tuple(values)


def _archive_pages(
    archiver: TushareBulkArchiver,
    api_name: str,
    params: dict[str, Any],
    *,
    optional: bool = False,
    log_complete: bool = True,
) -> list[ArchivedPage]:
    try:
        pages = list(
            archiver.archive_paginated(
                api_name,
                params,
                page_size=API_PAGE_SIZES.get(api_name, 1000),
            )
        )
    except TushareApiError as exc:
        if not optional:
            raise
        _log("optional_api_unavailable", api=api_name, code=exc.code)
        return []
    if log_complete:
        _log(
            "dataset_complete",
            api=api_name,
            params=params,
            pages=len(pages),
            rows=sum(page.row_count for page in pages),
            resumed=sum(page.resumed for page in pages),
        )
    return pages


def _reference(
    archiver: TushareBulkArchiver, start_date: str, end_date: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    stock_codes: set[str] = set()
    trading_days: set[str] = set()
    for api_name, params in REFERENCE_JOBS:
        pages = _archive_pages(archiver, api_name, params, optional=api_name == "stock_company")
        if api_name == "stock_basic":
            for page in pages:
                stock_codes.update(
                    str(row["ts_code"]) for row in archiver.read_items(page) if row.get("ts_code")
                )
    for exchange in ("SSE", "SZSE"):
        pages = _archive_pages(
            archiver,
            "trade_cal",
            {"exchange": exchange, "start_date": start_date, "end_date": end_date},
        )
        for page in pages:
            trading_days.update(
                str(row["cal_date"])
                for row in archiver.read_items(page)
                if str(row.get("is_open")) == "1" and row.get("cal_date")
            )
    for market in INDEX_MARKETS:
        _archive_pages(archiver, "index_basic", {"market": market}, optional=True)
    if not trading_days:
        raise RuntimeError("trading calendar contains no open sessions")
    return tuple(sorted(stock_codes)), tuple(sorted(trading_days))


def _market(
    archiver: TushareBulkArchiver,
    start_date: str,
    end_date: str,
    trading_days: tuple[str, ...],
    workers: int,
) -> None:
    """Use one request per session because compatible proxies may cap global offsets."""
    for api_name in MARKET_APIS:
        row_count = 0
        page_count = 0
        resumed_count = 0
        selected_days = tuple(day for day in trading_days if start_date <= day <= end_date)
        started_at = time.monotonic()

        def download_session(trade_date: str, current_api: str = api_name) -> list[ArchivedPage]:
            return _archive_pages(
                archiver,
                current_api,
                {"trade_date": trade_date},
                optional=current_api in {"limit_list_d", "moneyflow"},
                log_complete=False,
            )

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tushare") as executor:
            futures = [executor.submit(download_session, day) for day in selected_days]
            for number, future in enumerate(as_completed(futures), start=1):
                pages = future.result()
                row_count += sum(page.row_count for page in pages)
                page_count += len(pages)
                resumed_count += sum(page.resumed for page in pages)
                if number % 100 == 0 or number == len(selected_days):
                    _log(
                        "market_progress",
                        api=api_name,
                        sessions=number,
                        total_sessions=len(selected_days),
                        rows=row_count,
                        workers=workers,
                        checkpoint=archiver.state.counts(),
                        **_progress_timing(
                            started_at=started_at,
                            completed=number,
                            total=len(selected_days),
                        ),
                    )
        _log(
            "dataset_complete",
            api=api_name,
            params={
                "start_date": start_date,
                "end_date": end_date,
                "partition": "trade_date",
            },
            pages=page_count,
            rows=row_count,
            resumed=resumed_count,
            workers=workers,
            **_progress_timing(
                started_at=started_at,
                completed=len(selected_days),
                total=len(selected_days),
            ),
        )


def _financials(
    archiver: TushareBulkArchiver,
    stock_codes: Iterable[str],
    workers: int,
) -> None:
    """Archive complete statement history per stock.

    Tushare-compatible proxies may require ``ts_code`` for financial endpoints and
    reject cross-sectional requests containing only ``period``.  A listed company
    has far fewer rows than the API page limits, so stock partitioning is both
    complete and substantially cheaper than stock-by-quarter partitioning.
    """
    selected_codes = tuple(stock_codes)
    if not selected_codes:
        raise RuntimeError("financial backfill requires at least one stock code")

    for api_name in FINANCIAL_APIS:
        row_count = 0
        page_count = 0
        resumed_count = 0
        started_at = time.monotonic()

        def download_stock(ts_code: str, current_api: str = api_name) -> list[ArchivedPage]:
            return _archive_pages(
                archiver,
                current_api,
                {"ts_code": ts_code},
                optional=current_api in {"forecast", "express", "disclosure_date"},
                log_complete=False,
            )

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tushare") as executor:
            futures = [executor.submit(download_stock, code) for code in selected_codes]
            for number, future in enumerate(as_completed(futures), start=1):
                pages = future.result()
                row_count += sum(page.row_count for page in pages)
                page_count += len(pages)
                resumed_count += sum(page.resumed for page in pages)
                if number % 100 == 0 or number == len(selected_codes):
                    _log(
                        "financial_progress",
                        api=api_name,
                        stocks=number,
                        total_stocks=len(selected_codes),
                        rows=row_count,
                        workers=workers,
                        checkpoint=archiver.state.counts(),
                        **_progress_timing(
                            started_at=started_at,
                            completed=number,
                            total=len(selected_codes),
                        ),
                    )
        _log(
            "dataset_complete",
            api=api_name,
            params={"partition": "ts_code", "history": "complete"},
            pages=page_count,
            rows=row_count,
            resumed=resumed_count,
            workers=workers,
            **_progress_timing(
                started_at=started_at,
                completed=len(selected_codes),
                total=len(selected_codes),
            ),
        )


def _corporate_actions(
    archiver: TushareBulkArchiver, stock_codes: Iterable[str], workers: int
) -> None:
    selected_codes = tuple(stock_codes)
    started_at = time.monotonic()

    def download_stock(ts_code: str) -> None:
        for api_name in ("dividend", "share_float"):
            _archive_pages(
                archiver,
                api_name,
                {"ts_code": ts_code},
                optional=True,
                log_complete=False,
            )

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tushare") as executor:
        futures = [executor.submit(download_stock, code) for code in selected_codes]
        for number, future in enumerate(as_completed(futures), start=1):
            future.result()
            if number % 100 == 0 or number == len(selected_codes):
                _log(
                    "corporate_progress",
                    stocks=number,
                    total_stocks=len(selected_codes),
                    workers=workers,
                    checkpoint=archiver.state.counts(),
                    **_progress_timing(
                        started_at=started_at,
                        completed=number,
                        total=len(selected_codes),
                    ),
                )


def _indices(archiver: TushareBulkArchiver, start_date: str, end_date: str) -> None:
    for index_code in BENCHMARKS:
        _archive_pages(
            archiver,
            "index_daily",
            {"ts_code": index_code, "start_date": start_date, "end_date": end_date},
        )
        _archive_pages(
            archiver,
            "index_dailybasic",
            {"ts_code": index_code, "start_date": start_date, "end_date": end_date},
            optional=True,
        )
    for index_code in WEIGHTED_BENCHMARKS:
        for month_start, month_end in _months(max(start_date, "20050101"), end_date):
            _archive_pages(
                archiver,
                "index_weight",
                {
                    "index_code": index_code,
                    "start_date": month_start,
                    "end_date": month_end,
                },
                optional=True,
            )


def _industries(archiver: TushareBulkArchiver) -> None:
    codes: set[str] = set()
    for level in ("L1", "L2", "L3"):
        pages = _archive_pages(
            archiver,
            "index_classify",
            {"level": level, "src": "SW2021"},
            optional=True,
        )
        for page in pages:
            codes.update(
                str(row["index_code"]) for row in archiver.read_items(page) if row.get("index_code")
            )
    for code in sorted(codes):
        _archive_pages(archiver, "index_member", {"index_code": code}, optional=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable immutable Tushare history archive")
    parser.add_argument(
        "--stages",
        default="reference,market,financials,indices,industries,corporate",
        help="comma-separated stages",
    )
    parser.add_argument("--start-date", default="19900101")
    parser.add_argument("--end-date", default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--state", type=Path, default=Path("artifacts/tushare-backfill/state.sqlite3")
    )
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--min-free-gb", type=float, default=20)
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def _load_local_env(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


def _validate_date(value: str, *, field_name: str) -> None:
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise SystemExit(f"{field_name} must use YYYYMMDD") from exc


def _preflight(raw_root: Path, minimum_free_gb: float) -> dict[str, float]:
    raw_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(raw_root)
    free_gb = usage.free / 1024**3
    if free_gb < minimum_free_gb:
        raise SystemExit(
            f"only {free_gb:.1f} GiB free; at least {minimum_free_gb:.1f} GiB is required"
        )
    return {"free_gb": round(free_gb, 2), "minimum_free_gb": minimum_free_gb}


def _raw_size_bytes(raw_root: Path) -> int:
    return sum(path.stat().st_size for path in raw_root.rglob("*") if path.is_file())


def _write_report(
    path: Path,
    *,
    status: str,
    stages: tuple[str, ...],
    start_date: str,
    end_date: str,
    checkpoint: dict[str, int],
    raw_root: Path,
    error: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "aquant.tushare-backfill-report.v1",
        "status": status,
        "generated_at": datetime.now().astimezone().isoformat(),
        "stages": stages,
        "start_date": start_date,
        "end_date": end_date,
        "checkpoint": checkpoint,
        "raw_root": str(raw_root.resolve()),
        "raw_bytes": _raw_size_bytes(raw_root),
        "error": error,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = _parse_args()
    _load_local_env()
    token = os.environ.get("TUSHARE_TOKEN", "")
    endpoint = os.environ.get("TUSHARE_ENDPOINT", "https://api.tushare.pro")
    if not token:
        raise SystemExit("TUSHARE_TOKEN is required in .env or the process environment")
    _validate_date(args.start_date, field_name="start-date")
    _validate_date(args.end_date, field_name="end-date")
    if args.start_date > args.end_date:
        raise SystemExit("start-date cannot be after end-date")
    if not 1 <= args.workers <= 16:
        raise SystemExit("workers must be between 1 and 16")
    stages = tuple(stage.strip() for stage in args.stages.split(",") if stage.strip())
    unknown = set(stages) - {
        "reference",
        "market",
        "financials",
        "indices",
        "industries",
        "corporate",
    }
    if unknown:
        raise SystemExit(f"unknown stages: {sorted(unknown)}")
    disk = _preflight(args.raw_root, args.min_free_gb)
    report_path = args.state.parent / "report.json"
    lock_path = args.state.parent / "backfill.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another Tushare backfill process is already running") from exc
        archiver = TushareBulkArchiver(
            token=token,
            endpoint=endpoint,
            raw_root=args.raw_root,
            state_path=args.state,
            timeout_seconds=args.timeout,
            minimum_interval_seconds=args.interval,
            max_attempts=args.max_attempts,
        )
        _log(
            "backfill_started",
            stages=stages,
            start=args.start_date,
            end=args.end_date,
            disk=disk,
            workers=args.workers,
        )
        if args.status_only:
            _log("backfill_status", checkpoint=archiver.state.counts(), report=str(report_path))
            archiver.state.close()
            return 0
        stock_codes: tuple[str, ...] = ()
        trading_days: tuple[str, ...] = ()
        try:
            if (
                "reference" in stages
                or "market" in stages
                or "financials" in stages
                or "corporate" in stages
            ):
                stock_codes, trading_days = _reference(archiver, args.start_date, args.end_date)
                _log(
                    "reference_complete",
                    stocks=len(stock_codes),
                    trading_days=len(trading_days),
                )
            if "market" in stages:
                _market(
                    archiver,
                    args.start_date,
                    args.end_date,
                    trading_days,
                    args.workers,
                )
            if "financials" in stages:
                _financials(archiver, stock_codes, args.workers)
            if "indices" in stages:
                _indices(archiver, args.start_date, args.end_date)
            if "industries" in stages:
                _industries(archiver)
            if "corporate" in stages:
                _corporate_actions(archiver, stock_codes, args.workers)
            checkpoint = archiver.state.counts()
            _write_report(
                report_path,
                status="COMPLETED",
                stages=stages,
                start_date=args.start_date,
                end_date=args.end_date,
                checkpoint=checkpoint,
                raw_root=args.raw_root,
            )
            _log("backfill_complete", checkpoint=checkpoint, report=str(report_path))
        except Exception as exc:
            checkpoint = archiver.state.counts()
            _write_report(
                report_path,
                status="FAILED_RESUMABLE",
                stages=stages,
                start_date=args.start_date,
                end_date=args.end_date,
                checkpoint=checkpoint,
                raw_root=args.raw_root,
                error=f"{type(exc).__name__}: {exc}",
            )
            _log("backfill_paused", checkpoint=checkpoint, report=str(report_path))
            raise
        finally:
            archiver.state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
