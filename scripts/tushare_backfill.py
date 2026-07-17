import argparse
import fcntl
import json
import os
import shutil
from collections.abc import Iterable
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
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def _quarter_ends(start_date: str, end_date: str) -> tuple[str, ...]:
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    values = []
    for year in range(start_year, end_year + 1):
        for suffix in ("0331", "0630", "0930", "1231"):
            value = f"{year}{suffix}"
            if start_date <= value <= end_date:
                values.append(value)
    return tuple(values)


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
    _log(
        "dataset_complete",
        api=api_name,
        params=params,
        pages=len(pages),
        rows=sum(page.row_count for page in pages),
        resumed=sum(page.resumed for page in pages),
    )
    return pages


def _reference(archiver: TushareBulkArchiver, start_date: str, end_date: str) -> tuple[str, ...]:
    stock_codes: set[str] = set()
    for api_name, params in REFERENCE_JOBS:
        pages = _archive_pages(archiver, api_name, params, optional=api_name == "stock_company")
        if api_name == "stock_basic":
            for page in pages:
                stock_codes.update(
                    str(row["ts_code"]) for row in archiver.read_items(page) if row.get("ts_code")
                )
    for exchange in ("SSE", "SZSE"):
        _archive_pages(
            archiver,
            "trade_cal",
            {"exchange": exchange, "start_date": start_date, "end_date": end_date},
        )
    for market in INDEX_MARKETS:
        _archive_pages(archiver, "index_basic", {"market": market}, optional=True)
    return tuple(sorted(stock_codes))


def _market(archiver: TushareBulkArchiver, start_date: str, end_date: str) -> None:
    for api_name in MARKET_APIS:
        _archive_pages(
            archiver,
            api_name,
            {"start_date": start_date, "end_date": end_date},
            optional=api_name in {"limit_list_d", "moneyflow"},
        )


def _financials(archiver: TushareBulkArchiver, start_date: str, end_date: str) -> None:
    for period in _quarter_ends(start_date, end_date):
        for api_name in FINANCIAL_APIS:
            _archive_pages(
                archiver,
                api_name,
                {"period" if api_name != "disclosure_date" else "end_date": period},
                optional=api_name in {"forecast", "express", "disclosure_date"},
            )


def _corporate_actions(archiver: TushareBulkArchiver, stock_codes: Iterable[str]) -> None:
    for number, ts_code in enumerate(stock_codes, start=1):
        for api_name in ("dividend", "share_float"):
            _archive_pages(archiver, api_name, {"ts_code": ts_code}, optional=True)
        if number % 100 == 0:
            _log("corporate_progress", stocks=number, checkpoint=archiver.state.counts())


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
    parser.add_argument("--interval", type=float, default=0.15)
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
        )
        if args.status_only:
            _log("backfill_status", checkpoint=archiver.state.counts(), report=str(report_path))
            archiver.state.close()
            return 0
        stock_codes: tuple[str, ...] = ()
        try:
            if "reference" in stages or "corporate" in stages:
                stock_codes = _reference(archiver, args.start_date, args.end_date)
                _log("reference_complete", stocks=len(stock_codes))
            if "market" in stages:
                _market(archiver, args.start_date, args.end_date)
            if "financials" in stages:
                _financials(archiver, args.start_date, args.end_date)
            if "indices" in stages:
                _indices(archiver, args.start_date, args.end_date)
            if "industries" in stages:
                _industries(archiver)
            if "corporate" in stages:
                _corporate_actions(archiver, stock_codes)
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
