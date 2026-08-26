#!/usr/bin/env python3
import argparse
import json
from datetime import date, datetime
from pathlib import Path

import pyarrow.dataset as ds  # type: ignore[import-untyped]

from aquant.data.history import TushareHistoryCatalog
from aquant.data.industry import IndustryPITPublisher


def _date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYYMMDD") from exc


def _calendar(release_dir: Path) -> tuple[date, ...]:
    paths = tuple((release_dir / "dataset=trade_cal").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError("history release trade calendar parquet is missing")
    dataset = ds.dataset([str(path) for path in paths], format="parquet")
    table = dataset.to_table(columns=["cal_date", "is_open"])
    return tuple(sorted({row["cal_date"] for row in table.to_pylist() if row["is_open"]}))


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a Standard/PIT industry release")
    parser.add_argument("--start-date", type=_date, required=True)
    parser.add_argument("--end-date", type=_date, required=True)
    parser.add_argument("--classification-system", default="SW2014,SW2021")
    parser.add_argument("--industry-level", default="L1,L2,L3")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--data-release-id", required=True)
    parser.add_argument("--raw-state", type=Path, required=True)
    parser.add_argument("--history-release-dir", type=Path, required=True)
    parser.add_argument("--standard-root", type=Path, default=Path("data/standard"))
    parser.add_argument("--code-version", default="working-tree")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    systems = tuple(
        dict.fromkeys(item.strip().upper() for item in args.classification_system.split(","))
    )
    levels = tuple(dict.fromkeys(item.strip().upper() for item in args.industry_level.split(",")))
    if set(systems) - {"SW2014", "SW2021"}:
        parser.error("classification-system must contain only SW2014/SW2021")
    if set(levels) != {"L1", "L2", "L3"}:
        parser.error("the first PIT release requires L1,L2,L3 hierarchy")
    if not 1 <= args.max_workers <= 2:
        parser.error("max-workers must be between 1 and 2")
    if args.force:
        parser.error("--force cannot overwrite an immutable verified release")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "release_id": args.release_id,
                    "data_release_id": args.data_release_id,
                    "classification_systems": systems,
                    "industry_levels": levels,
                    "resume": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    try:
        result = IndustryPITPublisher(
            catalog=TushareHistoryCatalog(args.raw_state),
            standard_root=args.standard_root,
            release_id=args.release_id,
            data_release_id=args.data_release_id,
            trading_days=_calendar(args.history_release_dir),
            start_date=args.start_date,
            end_date=args.end_date,
            systems=systems,
            code_version=args.code_version,
        ).publish()
        print(
            json.dumps(
                {
                    "status": result.status,
                    "release_directory": str(result.release_directory),
                    "manifest": str(result.manifest_path),
                    "content_hash": result.content_hash,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
