import argparse
import fcntl
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from aquant.data.catalog import DataReleasePublisher
from aquant.data.daily_update import (
    DailyUpdateMode,
    UpdateJob,
    UpdateJobResult,
    announcement_dates,
    archive_job,
    build_update_plan,
    next_release_id,
    raw_batch_ids,
    result_to_dict,
    select_market_dates,
    validate_update_results,
)
from aquant.data.ingestion import TushareBulkArchiver
from aquant.data.normalization.standard_store import DailyBarParquetStore
from aquant.data.quality import QualityIssue, QualityReport, Severity

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _log(event: str, **payload: object) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "timestamp": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
                **payload,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed Tushare close and next-morning incremental updater"
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in DailyUpdateMode],
        default=DailyUpdateMode.CLOSE.value,
    )
    parser.add_argument("--as-of-date", help="Shanghai calendar date in YYYYMMDD")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--morning-lookback-sessions", type=int, default=3)
    parser.add_argument("--calendar-lookback-days", type=int, default=45)
    parser.add_argument("--min-free-gb", type=float, default=5)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--standard-root", type=Path, default=Path("data/standard"))
    parser.add_argument("--release-root", type=Path, default=Path("data/releases"))
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/daily-update"),
    )
    parser.add_argument("--force-refresh", action="store_true")
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


def _as_of_date(value: str | None) -> date:
    if value is None:
        return datetime.now(SHANGHAI).date()
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise SystemExit("as-of-date must use YYYYMMDD") from exc


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.workers <= 16:
        raise SystemExit("workers must be between 1 and 16")
    if args.interval < 0:
        raise SystemExit("interval must be non-negative")
    if args.timeout <= 0 or args.max_attempts <= 0:
        raise SystemExit("timeout and max-attempts must be positive")
    if args.morning_lookback_sessions <= 0 or args.calendar_lookback_days < 14:
        raise SystemExit("lookback configuration is too small")
    if args.min_free_gb < 0:
        raise SystemExit("min-free-gb must be non-negative")


def _run_directory(
    root: Path,
    *,
    as_of: date,
    mode: DailyUpdateMode,
    force_refresh: bool,
) -> Path:
    base = f"{as_of.strftime('%Y%m%d')}-{mode.value}"
    if force_refresh:
        base = f"{base}-{datetime.now(SHANGHAI).strftime('%H%M%S')}"
    return root / "runs" / base


def _preflight(raw_root: Path, minimum_free_gb: float) -> dict[str, float]:
    raw_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(raw_root).free / 1024**3
    if free_gb < minimum_free_gb:
        raise SystemExit(
            f"only {free_gb:.1f} GiB free; at least {minimum_free_gb:.1f} GiB is required"
        )
    return {"free_gb": round(free_gb, 2), "minimum_free_gb": minimum_free_gb}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)


def _calendar_result(
    archiver: TushareBulkArchiver,
    *,
    as_of: date,
    lookback_days: int,
) -> UpdateJobResult:
    return archive_job(
        archiver,
        UpdateJob(
            "trade_cal",
            {
                "exchange": "SSE",
                "start_date": (as_of - timedelta(days=lookback_days)).strftime("%Y%m%d"),
                "end_date": as_of.strftime("%Y%m%d"),
            },
            "calendar",
            required=True,
            expect_rows=True,
        ),
    )


def _execute_plan(
    archiver: TushareBulkArchiver,
    jobs: tuple[UpdateJob, ...],
    *,
    workers: int,
) -> tuple[UpdateJobResult, ...]:
    results: list[UpdateJobResult] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tushare-daily") as executor:
        futures = {executor.submit(archive_job, archiver, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            _log(
                "daily_update_job",
                api=result.job.api_name,
                group=result.job.group,
                status="FAILED" if result.error else "COMPLETED",
                rows=result.row_count,
                completed_jobs=completed,
                total_jobs=len(jobs),
                error=result.error,
            )
    result_order = {
        (job.api_name, json.dumps(dict(job.params), sort_keys=True)): index
        for index, job in enumerate(jobs)
    }
    return tuple(
        sorted(
            results,
            key=lambda item: result_order[
                (item.job.api_name, json.dumps(dict(item.job.params), sort_keys=True))
            ],
        )
    )


def _calendar_rows(
    archiver: TushareBulkArchiver,
    result: UpdateJobResult,
) -> tuple[dict[str, Any], ...]:
    if result.error is not None:
        raise RuntimeError(result.error)
    rows: list[dict[str, Any]] = []
    for page in result.pages:
        rows.extend(archiver.read_items(page))
    if not rows:
        raise RuntimeError("trading calendar returned no rows")
    return tuple(rows)


def _batch_index(
    *,
    as_of: date,
    mode: DailyUpdateMode,
    market_dates: tuple[date, ...],
    announcement_check_dates: tuple[date, ...],
    results: tuple[UpdateJobResult, ...],
) -> dict[str, Any]:
    return {
        "schema_version": "aquant.tushare-daily-index.v1",
        "generated_at": datetime.now(SHANGHAI).isoformat(),
        "as_of_date": as_of.isoformat(),
        "mode": mode.value,
        "market_dates": [value.isoformat() for value in market_dates],
        "announcement_check_dates": [value.isoformat() for value in announcement_check_dates],
        "jobs": [result_to_dict(result) for result in results],
    }


def _status(root: Path) -> int:
    found = False
    for mode in DailyUpdateMode:
        path = root / f"latest-{mode.value}.json"
        if not path.is_file():
            _log("daily_update_status", mode=mode.value, status="NEVER_RUN")
            continue
        found = True
        payload = json.loads(path.read_text(encoding="utf-8"))
        _log(
            "daily_update_status",
            mode=mode.value,
            status=payload.get("status"),
            as_of_date=payload.get("as_of_date"),
            release_id=payload.get("release_id"),
            report=str(path),
        )
    return 0 if found else 1


def _append_error(
    report: QualityReport,
    *,
    code: str,
    key: str,
    message: str,
) -> QualityReport:
    return QualityReport(
        report.dataset,
        report.checked_records,
        (*report.issues, QualityIssue(Severity.ERROR, code, key, message)),
    )


def _write_run_report(
    path: Path,
    *,
    status: str,
    as_of: date,
    mode: DailyUpdateMode,
    market_dates: tuple[date, ...],
    announcement_check_dates: tuple[date, ...],
    release_id: str | None,
    batch_index_path: Path | None,
    quality_report_path: Path | None,
    standard_data_path: Path | None,
    checkpoint: dict[str, int],
    error: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "aquant.tushare-daily-report.v1",
        "status": status,
        "generated_at": datetime.now(SHANGHAI).isoformat(),
        "as_of_date": as_of.isoformat(),
        "mode": mode.value,
        "market_dates": [value.isoformat() for value in market_dates],
        "announcement_check_dates": [value.isoformat() for value in announcement_check_dates],
        "release_id": release_id,
        "batch_index": str(batch_index_path) if batch_index_path else None,
        "quality_report": str(quality_report_path) if quality_report_path else None,
        "standard_data": str(standard_data_path) if standard_data_path else None,
        "checkpoint": checkpoint,
        "error": error,
    }
    _write_json(path, payload)
    return payload


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    if args.status_only:
        return _status(args.artifacts_root)

    _load_local_env()
    token = os.environ.get("TUSHARE_TOKEN", "")
    endpoint = os.environ.get("TUSHARE_ENDPOINT", "https://api.tushare.pro")
    if not token:
        raise SystemExit("TUSHARE_TOKEN is required in .env or the process environment")

    mode = DailyUpdateMode(args.mode)
    as_of = _as_of_date(args.as_of_date)
    run_dir = _run_directory(
        args.artifacts_root,
        as_of=as_of,
        mode=mode,
        force_refresh=args.force_refresh,
    )
    report_path = run_dir / "report.json"
    if report_path.is_file() and not args.force_refresh:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            _log(
                "daily_update_already_complete",
                mode=mode.value,
                as_of_date=as_of.isoformat(),
                release_id=existing.get("release_id"),
                report=str(report_path),
            )
            return 0

    disk = _preflight(args.raw_root, args.min_free_gb)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.artifacts_root / "daily-update.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path = args.artifacts_root / f"latest-{mode.value}.json"
    state_path = run_dir / "state.sqlite3"
    checkpoint: dict[str, int] = {}

    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another Tushare daily update process is already running") from exc

        archiver = TushareBulkArchiver(
            token=token,
            endpoint=endpoint,
            raw_root=args.raw_root,
            state_path=state_path,
            timeout_seconds=args.timeout,
            minimum_interval_seconds=args.interval,
            max_attempts=args.max_attempts,
        )
        _log(
            "daily_update_started",
            mode=mode.value,
            as_of_date=as_of.isoformat(),
            workers=args.workers,
            interval_seconds=args.interval,
            disk=disk,
            run_dir=str(run_dir),
        )
        market_dates: tuple[date, ...] = ()
        announcement_check_dates: tuple[date, ...] = ()
        batch_index_path: Path | None = None
        quality_report_path: Path | None = None
        standard_data_path: Path | None = None
        try:
            calendar = _calendar_result(
                archiver,
                as_of=as_of,
                lookback_days=args.calendar_lookback_days,
            )
            rows = _calendar_rows(archiver, calendar)
            market_dates = select_market_dates(
                rows,
                as_of_date=as_of,
                mode=mode,
                morning_lookback_sessions=args.morning_lookback_sessions,
            )
            announcement_check_dates = announcement_dates(
                as_of_date=as_of,
                market_dates=market_dates,
                mode=mode,
            )
            plan = build_update_plan(
                market_dates=market_dates,
                announcement_check_dates=announcement_check_dates,
            )
            _log(
                "daily_update_plan",
                mode=mode.value,
                market_dates=[value.isoformat() for value in market_dates],
                announcement_check_dates=[value.isoformat() for value in announcement_check_dates],
                jobs=len(plan),
            )
            results = (calendar, *_execute_plan(archiver, plan, workers=args.workers))
            batch_index_path = run_dir / "batch-index.json"
            _write_json(
                batch_index_path,
                _batch_index(
                    as_of=as_of,
                    mode=mode,
                    market_dates=market_dates,
                    announcement_check_dates=announcement_check_dates,
                    results=results,
                ),
            )
            validation = validate_update_results(
                archiver,
                results,
                market_dates=market_dates,
            )
            quality_report = validation.report
            quality_report_path = run_dir / "quality-report.json"
            _write_text(quality_report_path, quality_report.to_json())
            release_id = next_release_id(args.release_root, as_of)
            publisher = DataReleasePublisher(args.release_root)

            if not quality_report.passed:
                with suppress(ValueError):
                    publisher.publish(
                        release_id,
                        data_files=(batch_index_path,),
                        reports=(quality_report,),
                    )
                checkpoint = archiver.state.counts()
                report = _write_run_report(
                    report_path,
                    status="QUARANTINED",
                    as_of=as_of,
                    mode=mode,
                    market_dates=market_dates,
                    announcement_check_dates=announcement_check_dates,
                    release_id=str(release_id),
                    batch_index_path=batch_index_path,
                    quality_report_path=quality_report_path,
                    standard_data_path=None,
                    checkpoint=checkpoint,
                    error="quality gate blocked data release",
                )
                _write_json(latest_path, report)
                _log(
                    "daily_update_quarantined",
                    release_id=str(release_id),
                    report=str(report_path),
                )
                return 2

            data_files: list[Path] = [batch_index_path]
            if validation.bars:
                daily_pages = tuple(
                    page
                    for result in results
                    if result.job.api_name == "daily"
                    for page in result.pages
                )
                try:
                    raw_ids = tuple(UUID(value) for value in raw_batch_ids(daily_pages))
                    standard = DailyBarParquetStore(args.standard_root).write(
                        validation.bars,
                        provider="tushare",
                        raw_batch_ids=raw_ids,
                        standardized_at=datetime.now(UTC),
                    )
                except Exception as exc:
                    quality_report = _append_error(
                        quality_report,
                        code="STANDARDIZATION_FAILED",
                        key="bars_1d",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                    _write_text(quality_report_path, quality_report.to_json())
                    with suppress(ValueError):
                        publisher.publish(
                            release_id,
                            data_files=(batch_index_path,),
                            reports=(quality_report,),
                        )
                    checkpoint = archiver.state.counts()
                    report = _write_run_report(
                        report_path,
                        status="QUARANTINED",
                        as_of=as_of,
                        mode=mode,
                        market_dates=market_dates,
                        announcement_check_dates=announcement_check_dates,
                        release_id=str(release_id),
                        batch_index_path=batch_index_path,
                        quality_report_path=quality_report_path,
                        standard_data_path=None,
                        checkpoint=checkpoint,
                        error="standardization failed",
                    )
                    _write_json(latest_path, report)
                    _log(
                        "daily_update_quarantined",
                        release_id=str(release_id),
                        report=str(report_path),
                    )
                    return 2
                standard_data_path = standard.data_path
                data_files.extend((standard.data_path, standard.manifest_path))

            publisher.publish(
                release_id,
                data_files=tuple(data_files),
                reports=(quality_report,),
            )
            checkpoint = archiver.state.counts()
            report = _write_run_report(
                report_path,
                status="COMPLETED",
                as_of=as_of,
                mode=mode,
                market_dates=market_dates,
                announcement_check_dates=announcement_check_dates,
                release_id=str(release_id),
                batch_index_path=batch_index_path,
                quality_report_path=quality_report_path,
                standard_data_path=standard_data_path,
                checkpoint=checkpoint,
            )
            _write_json(latest_path, report)
            _log(
                "daily_update_complete",
                mode=mode.value,
                release_id=str(release_id),
                market_dates=[value.isoformat() for value in market_dates],
                checkpoint=checkpoint,
                report=str(report_path),
            )
            return 0
        except Exception as exc:
            checkpoint = archiver.state.counts()
            report = _write_run_report(
                report_path,
                status="FAILED_RESUMABLE",
                as_of=as_of,
                mode=mode,
                market_dates=market_dates,
                announcement_check_dates=announcement_check_dates,
                release_id=None,
                batch_index_path=batch_index_path,
                quality_report_path=quality_report_path,
                standard_data_path=standard_data_path,
                checkpoint=checkpoint,
                error=f"{type(exc).__name__}: {exc}",
            )
            _write_json(latest_path, report)
            _log(
                "daily_update_failed",
                mode=mode.value,
                checkpoint=checkpoint,
                report=str(report_path),
                error=report["error"],
            )
            raise
        finally:
            archiver.state.close()


if __name__ == "__main__":
    raise SystemExit(main())
