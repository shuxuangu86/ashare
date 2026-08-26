import argparse
import fcntl
import json
import shutil
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path

from aquant.data.history import (
    DEFAULT_HISTORY_DATASETS,
    TushareHistoryCatalog,
    TushareHistoryMaterializer,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize immutable Standard/PIT-ready history from local Tushare Raw"
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("artifacts/tushare-backfill/state.sqlite3"),
    )
    parser.add_argument("--standard-root", type=Path, default=Path("data/standard"))
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--start-date", default="20070104")
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_HISTORY_DATASETS),
        help="comma-separated dataset names",
    )
    parser.add_argument("--skip-raw-checksums", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=20)
    return parser.parse_args()


def _date(value: str, *, field: str) -> date:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise SystemExit(f"{field} must use YYYYMMDD") from exc


def _progress(event: str, payload: Mapping[str, object]) -> None:
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


def main() -> int:
    args = _parse_args()
    args.standard_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(args.standard_root).free / 1024**3
    if free_gb < args.min_free_gb:
        raise SystemExit(
            f"only {free_gb:.1f} GiB free; at least {args.min_free_gb:.1f} GiB is required"
        )
    datasets = tuple(value.strip() for value in args.datasets.split(",") if value.strip())
    materializer = TushareHistoryMaterializer(
        catalog=TushareHistoryCatalog(args.state),
        standard_root=args.standard_root,
        release_id=args.release_id,
        start_date=_date(args.start_date, field="start-date"),
        end_date=_date(args.end_date, field="end-date"),
        verify_raw_checksums=not args.skip_raw_checksums,
        progress=_progress,
    )
    lock_path = args.standard_root / f".{args.release_id}.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another history materialization is already running") from exc
        result = materializer.materialize(datasets)
    print(
        json.dumps(
            {
                "status": result.manifest.status,
                "release_id": result.manifest.release_id,
                "start_date": result.manifest.start_date,
                "end_date": result.manifest.end_date,
                "release_directory": str(result.release_directory),
                "datasets": [
                    {"dataset": dataset, "rows": rows}
                    for dataset, _checksum, rows in result.manifest.datasets
                ],
                "caveats": result.manifest.caveats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
