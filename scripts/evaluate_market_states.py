#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect materialized market-state evaluations")
    parser.add_argument("--artifact-directory", type=Path, default=Path("artifacts/regime_v1"))
    args = parser.parse_args()
    path = args.artifact_directory / "state_forward_return_evaluation.parquet"
    table = pq.read_table(path)
    print(
        json.dumps(
            {
                "rows": table.num_rows,
                "states": len(set(table.column("state_id").to_pylist())),
                "path": str(path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
