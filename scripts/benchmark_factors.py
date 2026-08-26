#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from aquant.domain.data_release import DataReleaseId
from aquant.factors.atomic import baseline_factor_library
from aquant.factors.data_loader import StandardPITFactorLoader
from aquant.factors.performance import benchmark_factor_workload

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_SCENARIOS = {
    "all_a_1y_50": (1, 50, False),
    "all_a_5y_50": (5, 50, False),
    "all_a_10y_73": (10, 73, False),
    "all_a_1y_500_candidates": (1, 500, False),
    "single_day_incremental_73": (0, 73, True),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark deterministic factor workloads")
    parser.add_argument("--scenario", choices=sorted(_SCENARIOS), required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--data-release-id", type=DataReleaseId, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor-store", type=Path)
    parser.add_argument("--code-version", default="working-tree")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    years, factor_count, incremental = _SCENARIOS[args.scenario]
    start_date = (
        args.end_date - timedelta(days=400)
        if incremental
        else date(args.end_date.year - years, args.end_date.month, args.end_date.day)
        + timedelta(days=1)
    )
    library = baseline_factor_library()
    factors = tuple(library[index % len(library)] for index in range(factor_count))
    fields = tuple(sorted({field for factor in factors for field in factor.spec.input_fields}))
    config = {
        "scenario": args.scenario,
        "start_date": start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "factor_count": factor_count,
        "fields": fields,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if args.dry_run:
        print(json.dumps({**config, "config_hash": config_hash}, indent=2))
        return 0
    panel = StandardPITFactorLoader(args.release_dir).load(
        fields=fields,
        start_date=start_date,
        end_date=args.end_date,
        as_of_time=datetime.combine(args.end_date, datetime.max.time(), _SHANGHAI),
        universe_id="all_a_share",
        data_release_id=args.data_release_id,
    )
    result = benchmark_factor_workload(
        scenario=args.scenario,
        panel=panel,
        factors=factors,
        data_release_id=str(args.data_release_id),
        code_version=args.code_version,
        config_hash=config_hash,
        factor_store=args.factor_store,
        output_date_count=1 if incremental else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".benchmark-",
        suffix=".json",
        dir=args.output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(result.payload(), stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(result.payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
