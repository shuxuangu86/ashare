#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import date, datetime
from pathlib import Path

from tushare_backfill import _archive_pages, _load_local_env

from aquant.data.ingestion import TushareBulkArchiver


def _write_report(release_id: str, result: dict[str, object]) -> Path:
    report = Path("artifacts/industry-backfill") / release_id / "report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Checkpointed Tushare industry classification and membership backfill"
    )
    parser.add_argument("--start-date", default="19900101")
    parser.add_argument("--end-date", default=date.today().strftime("%Y%m%d"))
    parser.add_argument(
        "--classification-system",
        default="SW2014,SW2021",
        help="comma-separated SW2014/SW2021 versions",
    )
    parser.add_argument("--industry-level", default="L1,L2,L3")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--state", type=Path, default=Path("artifacts/tushare-backfill/state.sqlite3")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--max-attempts", type=int, default=8)
    args = parser.parse_args()
    systems = tuple(
        dict.fromkeys(item.strip().upper() for item in args.classification_system.split(","))
    )
    levels = tuple(dict.fromkeys(item.strip().upper() for item in args.industry_level.split(",")))
    if set(systems) - {"SW2014", "SW2021"}:
        parser.error("classification-system must contain only SW2014/SW2021")
    if set(levels) - {"L1", "L2", "L3"}:
        parser.error("industry-level must contain only L1/L2/L3")
    if not 1 <= args.max_workers <= 2:
        parser.error("industry backfill max-workers must be between 1 and 2")
    if args.force:
        parser.error("--force cannot overwrite immutable completed Raw pages")
    plan = {
        "status": "DRY_RUN" if args.dry_run else "RUNNING",
        "release_id": args.release_id,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "classification_systems": systems,
        "industry_levels": levels,
        "max_workers": args.max_workers,
        "resume": True,
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0
    _load_local_env()
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        parser.error("TUSHARE_TOKEN is required")
    started = time.monotonic()
    archiver = TushareBulkArchiver(
        token=token,
        endpoint=os.environ.get("TUSHARE_ENDPOINT", "https://api.tushare.pro"),
        raw_root=args.raw_root,
        state_path=args.state,
        minimum_interval_seconds=args.interval,
        max_attempts=args.max_attempts,
    )
    try:
        codes: set[str] = set()
        classifications = 0
        resumed_pages = 0
        for system in systems:
            for level in levels:
                pages = _archive_pages(
                    archiver,
                    "index_classify",
                    {"level": level, "src": system},
                )
                resumed_pages += sum(page.resumed for page in pages)
                for page in pages:
                    rows = archiver.read_items(page)
                    classifications += len(rows)
                    codes.update(str(row["index_code"]) for row in rows if row.get("index_code"))
        membership_rows = 0
        failed = 0
        for position, code in enumerate(sorted(codes), start=1):
            try:
                pages = _archive_pages(
                    archiver,
                    "index_member",
                    {"index_code": code},
                    log_complete=False,
                )
                resumed_pages += sum(page.resumed for page in pages)
                membership_rows += sum(page.row_count for page in pages)
            except Exception:
                failed += 1
                raise
            if position % 50 == 0 or position == len(codes):
                print(
                    json.dumps(
                        {
                            "event": "industry_backfill_progress",
                            "completed_codes": position,
                            "total_codes": len(codes),
                            "membership_rows": membership_rows,
                            "failed": failed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        result = {
            "status": "PASS",
            "release_id": args.release_id,
            "classification_rows": classifications,
            "industry_codes": len(codes),
            "membership_rows": membership_rows,
            "resumed_pages": resumed_pages,
            "failed": failed,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        report = _write_report(args.release_id, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        result = {
            "status": "BLOCKED",
            "reason_code": "EXTERNAL_DATA_SOURCE_ACCESS_FAILED",
            "release_id": args.release_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "failed": 1,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "resume_supported": True,
        }
        report = _write_report(args.release_id, result)
        print(
            json.dumps({**result, "report": str(report)}, indent=2, sort_keys=True),
            flush=True,
        )
        return 2
    finally:
        archiver.state.close()


if __name__ == "__main__":
    raise SystemExit(main())
