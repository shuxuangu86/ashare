#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from aquant.factors.selection import converge_cached_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Converge an audited disk-backed factor evaluation"
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--maximum-distance", type=float, default=0.5)
    args = parser.parse_args()
    try:
        payload = converge_cached_evaluation(
            cache_root=args.cache_root,
            report_dir=args.report_dir,
            output=args.output,
            horizon=args.horizon,
            maximum_distance=args.maximum_distance,
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "compact_factor_count": payload["compact_factor_count"],
                    "content_hash": payload["content_hash"],
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if payload["status"] == "PASS" else 2
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
