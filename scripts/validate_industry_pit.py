#!/usr/bin/env python3
import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import yaml

from aquant.data.industry.quality import (
    IndustryQualityThresholds,
    validate_industry_release,
)


def _date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYYMMDD") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Standard/PIT industry release")
    parser.add_argument("--industry-release", type=Path, required=True)
    parser.add_argument("--history-release", type=Path, required=True)
    parser.add_argument("--start-date", type=_date, required=True)
    parser.add_argument("--end-date", type=_date, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/data/industry_pit_v1.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/data_quality/industry_pit"),
    )
    args = parser.parse_args()
    if args.start_date > args.end_date:
        parser.error("date range is invalid")
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    payload.pop("schema_version", None)
    if "market_exchanges" in payload:
        payload["market_exchanges"] = tuple(payload["market_exchanges"])
    try:
        thresholds = IndustryQualityThresholds(**payload)
        report = validate_industry_release(
            industry_release=args.industry_release,
            history_release=args.history_release,
            start_date=args.start_date,
            end_date=args.end_date,
            thresholds=thresholds,
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    destination = args.output_root / str(report["industry_release_id"])
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "quality.json"
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({**report, "report": str(output)}, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
