from __future__ import annotations

import argparse
from pathlib import Path

from aquant.factors.selection.library_v2 import (
    publish_feature_eligible_pool,
    run_l3_family_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-l3", action="store_true")
    args = parser.parse_args()
    publish_feature_eligible_pool(
        report_dir=args.report_dir,
        catalog_path=args.catalog,
        output_dir=args.output_dir,
    )
    if not args.skip_l3:
        run_l3_family_smoke(
            cache_root=args.cache_root,
            report_dir=args.report_dir,
            pool_path=args.output_dir / "feature_eligible_pool.json",
            output=args.output_dir / "l3_family_smoke.json",
        )


if __name__ == "__main__":
    main()
